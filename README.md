# mlx-fastframes-swift

Apple **FastFrames-S** (arXiv 2611.23226, `apple/ml-fastframes`) on MLX in Swift. It's an autoregressive two-step
text-to-video model: a frame-causal Wan2.1-1.3B DiT, a KV refinement network, and a causal ConvNeXt pixel decoder that
writes RGB directly. Output is 81 frames at 832×480 and 16 fps.

**Status: V2 (Swift core with parity gates), 2026-11-18.** There's no engine package yet; V3 adds that. The porting
evaluation and the V0 Python-MLX spike are in `mlxengine-video/WIP/fastframes-eval/`.

## Licence channel — read first

| Part | Licence | Where it may go |
|---|---|---|
| DiT weights (Causal Forcing++ `framewise-2step.pt`, `zhuhz22/Causal-Forcing`) | Apache-2.0 | anywhere |
| KV refiner + pixel decoder weights (`apple/FastFrames-S`) | **Apple Machine Learning Research Model License** — research only, non-commercial, revocable | **Gaea plugins only, opt-in and named-only** (AB-D-0259 under AB-D-0226 channel 2). Never a shipping app. No weight mirror without a separate operator decision. |
| This code | a translation of Apple's sample code (Apple Sample Code License) — see `NOTICE` | — |

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

## Layout

- `FastFramesMLX`: the engine-agnostic core.
  - `CausalDiT`: per-frame forward over a preallocated `[1, H, 21·1560, D]` KV cache, written in place.
  - `KVRefiner`
  - `PixelDecoder`: the depthwise 3×3×3 conv is written as 27 shifted multiply-adds, with no MLX conv kernel.
  - `FastFramesPipeline.rollout`
  - `FastFramesTextEncoder`: wan-core's umT5 with upstream's zero padding.
- `ff-smoke`: the gates and runs (`lanes`, `rollout`, `t5`, `generate`); see the header of `Sources/Smoke/main.swift`.
- `Tools/export_gates.py`: converts the V0 spike's numpy artefacts into the safetensors the gates read.
- `Tools/pymlx_rollout_fp32.py`: the fp32 Python-MLX rollout oracle, with a `--perturb` sensitivity probe.

The weights keep upstream's torch key names. Convert the DiT from the `.pt` with the spike's `ref_torch.py convert`.

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

## Not yet done

- V2b: quantized tiers with a perceptual gate. The refiner and decoder were trained against a bf16 DiT.
- V3: an MLXEngine `textToVideo` package, flagged non-permissive.
- V5: a Gaea plugin.
- Rolling cache past 81 frames.
- FF-L as a second variant.
