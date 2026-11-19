# mlx-fastframes-swift

Apple **FastFrames-S** (arXiv 2611.23226, `apple/ml-fastframes`) on MLX in Swift. It's an autoregressive two-step
text-to-video model: a frame-causal Wan2.1-1.3B DiT, a KV refinement network, and a causal ConvNeXt pixel decoder that
writes RGB directly. Output is 81 frames at 832×480 and 16 fps.

**Status: PROD (promoted 2026-11-19).** V2 parity gates and V3's MLXEngine `textToVideo` package are done. The
package passes the offline conformance gates and runs through the real `MLXServeEngine`. The converted DiT is published
as [`mlx-community/Causal-Forcing-PP-framewise-2step-bf16`](https://huggingface.co/mlx-community/Causal-Forcing-PP-framewise-2step-bf16),
so every weight source materializes on a fresh machine. It runs in [Gaea](https://github.com/xocialize-code/gaea) as the
named-only, opt-in research plugin
[`gaea-plugin-fastframes`](https://github.com/xocialize-code/gaea-plugin-fastframes). The porting evaluation and the
V0 Python-MLX spike are in `mlxengine-video/WIP/fastframes-eval/`.

## Licence channel — read first

| Part | Licence | Where it may go |
|---|---|---|
| DiT weights (Causal Forcing++ `framewise-2step.pt`, `zhuhz22/Causal-Forcing`) | Apache-2.0 | anywhere |
| KV refiner + pixel decoder weights (`apple/FastFrames-S`) | **Apple Machine Learning Research Model License** — research only, non-commercial, revocable | **Gaea plugins only, opt-in and named-only** (AB-D-0259 under AB-D-0226 channel 2). Never a shipping app. No weight mirror without a separate operator decision. |
| This code | a modified translation of Apple's sample code, under the Apple Sample Code License (`LICENSE`); see `NOTICE` | — |

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

## Layout

- `FastFramesMLX`: the engine-agnostic core.
  - `CausalDiT`: per-frame forward over a preallocated `[1, H, 21·1560, D]` KV cache, written in place.
  - `KVRefiner`
  - `PixelDecoder`: the depthwise 3×3×3 conv is written as 27 shifted multiply-adds, with no MLX conv kernel.
  - `FastFramesPipeline.rollout`
  - `FastFramesTextEncoder`: wan-core's umT5 with upstream's zero padding.
- `MLXFastFrames`: the MLXEngine package (`FastFramesPackage`, `FastFramesConfiguration`).
- `ff-smoke`: the gates and runs (`lanes`, `rollout`, `onestep`, `t5`, `generate`, `engine`); see the header of
  `Sources/Smoke/main.swift`.
- `Tools/export_gates.py`: converts the V0 spike's numpy artefacts into the safetensors the gates read.
- `Tools/pymlx_rollout_fp32.py`: the fp32 Python-MLX rollout oracle, with a `--perturb` sensitivity probe.
- `Tools/dump_oracle_kv.py`: the same oracle rollout plus its KV cache, the reference for `onestep`.

The weights keep upstream's torch key names. Convert the DiT from the `.pt` with the spike's `ref_torch.py convert`.

## Gates on mlx-swift 0.33.1 (2026-11-19, M5 Max 128 GB, macOS 27.3, release build)

The package takes the fleet fork `xocialize/mlx-swift` "0.32.4" ..< "0.34.0" (0.33.1 = upstream 0.33.0 with the
head-dim-split SDPA routes disabled, AB-D-0264) and mlx-engine-swift ≥ 0.82.1. On 0.33.1 the V2 rollout gates moved
while every per-step number held, so the gate set was redesigned (operator-approved, AB-R-1009). Logs:
`receipts/*_mlxswift0331.log`.

| Gate | Threshold | 0.33.1 result |
|---|---|---|
| fp32 lanes vs upstream (all 9) | rel ≤ 1e-4 | 1.08–2.84e-6 (unchanged from V2 within ±1e-7) |
| **one-step**, teacher-forced x0 vs the Python-MLX oracle, frames 0–13 (new) | rel ≤ 5e-5 per frame | worst 9.12e-6 (frame 0); frames 1–13: 1.23–1.87e-6 |
| fp32 rollout vs the Python-MLX fp32 oracle (now built on mlx 0.32.4), frames 0–13 | max rel ≤ **3e-3** (was 1e-3) | 1.79e-3 |
| bf16 rollout vs Python-MLX bf16, frames 0–2 | cos ≥ 0.997 | 0.998025 |
| bf16 rollout vs upstream torch bf16, frames 0–2 | cos ≥ **0.9965** (was 0.997), PSNR ≥ 30 dB | 0.998869 / 0.997932 / 0.996941; PSNR min 31.26 dB |
| umT5 vs upstream CPU bf16 | 11/11 tokens, cos ≥ 0.999 | 11/11, 0.999778 (identical on 0.32.4 with the same weights) |
| offline (`swift test`) | — | 17/17 |

**Why the thresholds moved.** The rollout is autoregressive and amplifies ulp-level backend differences. Teacher-forced
from the oracle's own KV cache, every frame's x0 agrees to 1.2–2.3e-6 (frame 0, 4 steps + the clean pass: ≤ 1.1e-5) on
mlx-swift 0.32.4 and 0.33.1 alike, the same spread Python mlx 0.31.2 and 0.32.4 show against each other. Over the
rollout, those two Python oracles drift 7e-4 apart by frame 13 on their own, and Swift lands at 6.0e-4 (0.32.4) to
2.3e-3 (0.33.1) depending on which backend it pairs with. So `onestep` holds the tight per-step bound, the fp32 rollout
limit sits above the measured oracle-vs-oracle spread, and the bf16-vs-torch floor (a torch oracle can't follow MLX
kernel changes) drops by 5e-4.

**Gate files** live beside the others in `/Volumes/DEV_ARCHIVE/fastframes-s-measure/gates/` (too big for git):
- `pymlx_fp32_corgi_mlx0324.safetensors` (105 MB, sha256 `e4929c4b…`): `MLX_ENABLE_TF32=0 python -I
  Tools/pymlx_rollout_fp32.py --spike-dir <spike> --dit cf-pp/framewise-2step-ema.safetensors --refiner
  fastframes-s/kv_refiner.safetensors --decoder fastframes-s/pixel_decoder.safetensors --inputs out/inputs_corgi.npz
  --frames 21`, Python 3.12.16, mlx 0.32.4. The same command on mlx 0.31.2 reproduces the V2 reference
  `pymlx_fp32_corgi.safetensors` byte for byte.
- `onestep_kv_corgi_mlx0324.safetensors` (8.1 GB, sha256 `8152b86d…`): `Tools/dump_oracle_kv.py`, same arguments
  with `--frames 14`, same environment.

## V2 results (M5 Max 128 GB, macOS 27.3, mlx-swift 0.32.3, release build)

Logs and sheets are in `receipts/`.

**Parity**

| Gate | Result |
|---|---|
| fp32 lanes vs upstream (torch CPU fp32, attention kept fp32): DiT flow, hidden, K cache; refiner φ, K′, V′, ΔK, ΔV; decoder | all **1.1–2.9e-6** relative L2 (tolerance 1e-4); the same as the Python port |
| fp32 full rollout vs the Python-MLX fp32 rollout, latent frames 0–13 | max relative error **6.0e-4** (gate 1e-3) |
| bf16 rollout vs upstream torch bf16, frames 0–2 | cosine 0.99900 / 0.99802 / 0.99723; pixel PSNR 31.7 dB minimum, 33.8 dB mean. Python-MLX scored 0.99912 / 0.99845 / 0.99749 and 32.3 dB here. |
| bf16 rollout vs Python-MLX bf16, frames 0–2 | cosine ≥ 0.99879 |
| umT5 (wan-core) vs upstream CPU bf16 prompt embedding | 11 of 11 tokens, cosine 0.99984 |

**Why the gates stop at frame 13.**
- From latent frame 14 on, this prompt and seed amplify any perturbation.
- The Python-MLX fp32 oracle, run against itself with the prompt scaled by (1 + 1e-6), agrees to ≤ 4e-4 through frame
  13, then drifts to 7–9% by frames 19–20 (`receipts/oracle_sensitivity_perturb1e-6.log`).
- Swift against Python lands in the same band: 12% at frame 20 in fp32, more in bf16.
- So late frames are judged by eye. `receipts/sheet_swift_vs_pymlx.png`: the same scene and light, with the dog's pose
  diverging after frame 60.

**Speed and memory (bf16; the first 3 runs reuse a warm Metal shader cache)**

| | Swift | V0 Python-MLX |
|---|---|---|
| Full rollout, 21 latent → 81 RGB frames | **10.57–10.68 s** | 11.03 s |
| Frames 1–20, mean per latent frame (frame 1 → frame 20) | 0.457 s (0.281 → 0.641) | 0.477 s (0.287 → 0.672) |
| KV refiner per frame | 0.039 s | 0.040 s |
| Pixel decode, whole clip | 0.30–0.34 s | 0.30–0.35 s |
| MLX peak | 12.36 GB | 12.39 GB |
| Process peak footprint | 18.8–19.0 GB | 18.8 GB |

**End to end** (`generate`: prompt → umT5 → evict → rollout → MP4)
- Car prompt, seed 7: a coherent 81-frame h264 clip. The road winds, and the car is tracked at golden hour
  (`receipts/sheet_car_seed7.png`).

**umT5 cost**
- wan-core upcasts the encoder to fp32, which peaks at 23.7 GB.
- Encoding takes 47.8 s from a cold file cache and 1.2 s warm.
- The process footprint peak of 35 GB in `generate` comes from that phase. V3 should load the encoder in bf16 or run it
  in its own phase.

## V3: the engine package

**Surface.** One `textToVideo` surface, `fastframes-s-t2v`, on mlx-engine-swift 0.81.0 (contract 1.66).
- `numFrames` can be 1–81 (default 81). Latent frames = ⌈(n + 3) / 4⌉, so shorter clips are proportionally faster
  and the KV cache is sized to the clip.
- `fps` sets playback (default 16, the model's native rate). `seed` sets the MLX noise (random if unset).
- Refused legibly with `unsupportedRequestFeature`:
  - other sizes;
  - `steps` other than 2, `guidanceScale` other than 1, any `negativePrompt` (the schedule is distilled, with no
    CFG);
  - image, audio, keyframe and adapter conditioning, and `includeAudio`.
- Cancellation and `RunProgress(.denoise)` happen after every latent frame.

**Licence declaration.**
- Weights: `apache2` (the DiT) plus `LicenseRef-Apple-AMLR` (Apple's add-ons).
- Port code: `LicenseRef-Apple-Sample-Code`.
- Each non-standard licence carries an informational `LicenseNotice` with its terms link, and the AMLR notice
  includes the required attribution sentence verbatim.
- Behaviour by host:
  - A `.blocking` permissive-only host refuses the package (tested).
  - The default advisory host registers it and records the AMLR advisory.
- The engine has no named constant for either licence yet; an ask to `mlx-engine` covers that and the
  eval-acknowledged list.

**Weights.** Four sources. Apple's files come from Apple's repo, never a mirror (AB-D-0259).

| Role | Repo | Status |
|---|---|---|
| dit | `mlx-community/Causal-Forcing-PP-framewise-2step-bf16` (`dit.safetensors`, `Tools/convert_dit.py`) | live since 2026-11-18 |
| addons | `apple/FastFrames-S` (`fastframes-s/*.safetensors`) | live |
| text-encoder | `mlx-community/Bernini-R-1.3B-bf16` (`t5_encoder.safetensors`, `config.json`) | live |
| tokenizer | `google/umt5-xxl` | live |

**umT5 runs in bf16** (`pipeline.py:148` does the same).
- Gate: cosine 0.99978 vs upstream, against 0.99984 for wan-core's fp32 upcast.
- Peak: 11.9 GB instead of 23.6 GB.
- Process footprint: 12.2 GB instead of 35.2 GB (`receipts/t5_dtypes.log`).

**Measured through MLXServeEngine** (`receipts/engine_*.log`, `receipts/sheet_engine_car_seed7.png`)

| | Result |
|---|---|
| 81 frames, warm: umT5 + rollout + decode + MP4 | **12.93 s** |
| 33 frames, cold, including prewarm (15.1 GB in 1.7 s) and load | 8.26 s |
| Phys floor after load | 6.86 GB |
| Lifetime peak | **16.08 GB** (the prompt phase: resident set plus umT5, evicted before denoising) |
| Declared footprint | resident 4.0 GB + activation 14.7 GB |

Offline gates (`swift test`, 17 tests): MAT, the licence posture, FIT, CAN-1 and CAN-3, request planning, the load
refusal, and the Codable form without host paths.

**Published DiT, verified from a fresh store** (`receipts/engine_fresh_store_dit.log`)
- The engine found the DiT missing, downloaded it from the Hub into an empty model store, and ran.
- The downloaded `dit.safetensors` SHA-256 is `ab6b1d50…4c197`, the same as the local conversion.
- `convert_dit.py`'s output is tensor-identical to the gated file.
- The clip's 81 decoded frames are identical to the local-DiT run.
- Total time 80.3 s, including the 2.8 GB download, load and run.

## Use

```swift
// .package(url: "https://github.com/xocialize/mlx-fastframes-swift", from: "0.1.0")
let id = try await engine.register(FastFramesPackage.registration, configuration: FastFramesConfiguration())
let clip = try await engine.run(T2VRequest(prompt: "a corgi surfing at sunset", numFrames: 81, seed: 0), package: id)
```

The engine downloads all four weight sources on first use. Hosts that must not load non-commercial weights should
register with `licenseEnforcement: .blocking`, which refuses this package.

## Not yet done

- V2b: quantized tiers with a perceptual gate. The refiner and decoder were trained against a bf16 DiT.
- `writeMP4` isn't byte-repeatable: VideoToolbox's H.264 encoder can choose different frame types for the same frames
  run to run (the decoded frames then differ by about 50 dB in the affected GOPs). Compare decoded frames, not bytes.
- Rolling cache past 81 frames.
- FF-L as a second variant.
