// ff-smoke — V2 gates and runs for the FastFrames-S Swift core.
//
//   lanes     fp32 component lanes vs upstream (torch CPU fp32, attention kept fp32): DiT flow / hidden / K cache,
//             refiner phi / K′ / V′, pixel decoder. Run with MLX_ENABLE_TF32=0.
//   rollout   bf16 rollout from the spike's shared noise table, compared per frame with the Python-MLX V0 rollout
//             (--pymlx; --dtype f32 with a pymlx_rollout_fp32.py oracle) and upstream's first 3 frames (--ref); times every frame; optional --out writes an MP4,
//             timing.json and the uint8 pixels.
//   onestep   teacher-forced one-frame steps (fp32; run with MLX_ENABLE_TF32=0): for each latent frame k < 14 the KV
//             cache is the Python-MLX oracle's own (--oracle, from Tools/dump_oracle_kv.py), Swift denoises frame k with
//             the table noise and writes its context; x0_k is gated against the oracle's at rel ≤ 5e-5 per frame.
//   t5        wan-core umT5 prompt encoding vs the spike's upstream (CPU bf16) prompt embedding.
//   generate  prompt → umT5 (evicted) → rollout → decode → MP4, with seeded MLX noise.
//   engine    FastFramesPackage through the REAL MLXServeEngine (register → run; the engine constructs and loads the
//             package, C13): --addons <dir> --wan-dir <t5 dir> --tokenizer <dir> --dit <file> --prompt … [--frames N]
//             [--seed S] [--warmup] [--out dir] [--store dir: fetch the DiT from the Hub into a fresh model store]. Reports the phys floor after load and the lifetime peak during the
//             run — the basis of the manifest's split footprint. One measured run per process.
//
// Weights: --dit (Causal Forcing++ framewise-2step generator_ema as safetensors), --refiner, --decoder.

import FastFramesMLX
import Foundation
import MLX
import MLXFastFrames
import MLXServeCore
import MLXToolKit

setvbuf(stdout, nil, _IONBF, 0)
let args = Args(CommandLine.arguments)
if let gbLimit = args.double("cache-limit-gb") { Memory.cacheLimit = Int(gbLimit * 1e9) }

func loadPipeline(_ dtype: DType) throws -> FastFramesPipeline {
    let t = now()
    let p = try FastFramesPipeline.load(
        dit: args.url("dit"), refiner: args.url("refiner"), decoder: args.url("decoder"), dtype: dtype)
    print(String(format: "loaded weights (%@) in %.2f s", "\(dtype)", now() - t))
    return p
}

var failures: [String] = []
func gate(_ name: String, _ ok: Bool, _ detail: String) {
    print("\(ok ? "PASS" : "FAIL")  \(name)  \(detail)")
    if !ok { failures.append(name) }
}

switch args.mode {
case "lanes":
    let tol = args.double("tol") ?? 1e-4
    let p = try loadPipeline(.float32)
    let inputs = try loadArrays(url: args.url("inputs"))
    let L = try loadArrays(url: args.url("lanes"))
    let emb = inputs["prompt_embeds"]![.newAxis]
    let x = inputs["noise"]![0][.newAxis]
    let tc = p.dit.timeConditioning(1000.0)
    let text = p.dit.textCache(emb)
    let kv = FrameKVCache(frames: 1, dtype: .float32)
    let (flow, hidden) = p.dit.forwardFrame(x, tc, text, kv, frame: 0)
    eval([flow, hidden] + kv.arrays)
    func lane(_ name: String, _ ref: MLXArray, _ got: MLXArray) {
        let (r, c) = compare(ref, got)
        gate(name, r < tol, String(format: "rel %.2e  cos %.9f", r, c))
    }
    lane("DiT flow", L["flow"]!.reshaped(flow.shape), flow)
    lane("DiT hidden (decoder input)", L["feat"]!.reshaped(hidden.shape), hidden)
    let nK = stacked(kv.k.map { $0[0].transposed(1, 0, 2).reshaped(FF.frameTokens, FF.dim) })
    lane("DiT K cache (30 layers)", L["nK"]!, nK)
    let e = p.refiner.embedTimestep(FF.refineTimestep)
    let stem = p.dit.patchEmbed(L["x0"]!.reshaped(1, FF.latentChannels, FF.latentH, FF.latentW))
    let phi = p.refiner.computePhi(stem, e)
    let (rk, rv) = p.refiner.repairAll(L["nK"]!, L["nV"]!, phi, e)
    eval(phi, rk, rv)
    lane("Refiner phi", L["phi"]!, phi)
    lane("Refiner K'", L["rk"]!, rk)
    lane("Refiner V'", L["rv"]!, rv)
    lane("Refiner dK = K'-K", L["rk"]! - L["nK"]!, rk - L["nK"]!)
    lane("Refiner dV = V'-V", L["rv"]! - L["nV"]!, rv - L["nV"]!)
    let eh = L["eh"]!
    let px = p.decoder(L["feat"]!, eh.reshaped(1, 1, 1, FF.dim))
    eval(px)
    lane("Pixel decoder (4 RGB frames)", L["px"]!.reshaped(px.shape), px)

case "rollout":
    let frames = args.int("frames", FF.numFrames)
    let runs = args.int("runs", 1)
    let dtype: DType = args.str("dtype") == "f32" ? .float32 : .bfloat16
    let p = try loadPipeline(dtype)
    let inputs = try loadArrays(url: args.url("inputs"))
    let emb = inputs["prompt_embeds"]![.newAxis]
    let noise = TableNoise(arrays: inputs, dtype: dtype)
    var result: RolloutResult!
    var report: [[String: Any]] = []
    var pixels: MLXArray!
    for run in 0..<runs {
        Memory.peakMemory = 0
        let t = now()
        result = p.rollout(promptEmbeds: emb, noise: noise, frames: frames, log: { print($0) })
        eval(result.latents, result.hidden, result.modulation)
        let rollS = now() - t
        let td = now()
        let px = p.decoder(result.hidden, result.modulation)[0, FF.droppedLeadingFrames...]
        pixels = ((clip(px, min: -1, max: 1) + 1) * 127.5).round().asType(.uint8).transposed(0, 2, 3, 1)
        eval(pixels)
        let decS = now() - td
        let peak = Memory.peakMemory
        print(String(format: "run %d: rollout %.2f s, decode %.2f s, MLX peak %@, footprint peak %@",
                     run, rollS, decS, gb(peak), gb(peakFootprint())))
        report.append([
            "rollout_s": rollS, "decode_s": decS, "mlx_peak_bytes": peak,
            "frames": result.timings.map {
                ["frame": $0.frame, "denoise_s": $0.denoiseSeconds, "context_s": $0.contextSeconds]
            },
        ])
    }
    if let pymlx = args.str("pymlx") {
        let ref = try loadArrays(url: URL(fileURLWithPath: pymlx))
        // The gate covers the trajectory's stable window only. From latent frame 14 on, this prompt/seed amplifies
        // any perturbation: the Python-MLX fp32 oracle against itself with the prompt scaled by (1 + 1e-6) agrees to
        // ≤ 4e-4 through frame 13, then drifts to 7–9 % by frames 19–20. Late frames are judged perceptually.
        //   fp32: max latent rel over frames 0..<stable (default 14) ≤ 3e-3 — the logic gate (reference: the mlx 0.32.4
        //         oracle, gates/pymlx_fp32_corgi_mlx0324.safetensors)
        //   bf16: min latent cos over frames 0..<stable (default 3) ≥ 0.997 — the backend-rounding band V0 measured
        // Why 3e-3, not the V2 1e-3 (AB-R-1009, 2026-11-19): the rollout drift is amplification, not error. Teacher-forced
        // from the oracle's own KV cache (`onestep`), every frame's x0 is within 1.2–2.3e-6 rel (frame 0: ≤ 1.1e-5) on
        // mlx-swift 0.32.4 and 0.33.1 alike, the same as Python mlx 0.31.2 vs 0.32.4 against each other — yet two Python
        // oracles (0.31.2 vs 0.32.4) drift 7e-4 apart by frame 13 on their own, and Swift lands at 6.0e-4–2.3e-3 by
        // frame 13 depending on which ulp-level backend it pairs with. The tight check is `onestep`; this one catches
        // trajectory-level logic errors.
        let fp32 = dtype == .float32
        let stable = min(frames, args.int("stable-frames", fp32 ? 14 : 3))
        print("— vs Python-MLX V0 (\(fp32 ? "fp32" : "bf16"), same noise; gate on frames 0..<\(stable)) —")
        var maxRel = 0.0
        var minCos = 1.0
        for f in 0..<frames {
            let (r, c) = compare(ref["latents"]![f], result.latents[0, f])
            if f < stable { maxRel = max(maxRel, r); minCos = min(minCos, c) }
            print(String(format: "  latent frame %2d  rel %.4f  cos %.6f%@", f, r, c, f < stable ? "" : "  (ungated)"))
        }
        let n = frames * 4 - FF.droppedLeadingFrames
        let ps = psnr(ref["pixels"]![..<n], pixels)
        let mean = ps.reduce(0, +) / Double(ps.count)
        print(String(format: "  pixels (%d frames)  PSNR min %.2f dB  mean %.2f dB", n, ps.min()!, mean))
        if fp32 {
            gate("rollout vs Python-MLX (fp32 logic)", maxRel <= 3e-3,
                 String(format: "max rel %.2e over frames 0..<%d", maxRel, stable))
        } else {
            gate("rollout vs Python-MLX (bf16 band)", minCos >= 0.997,
                 String(format: "min cos %.6f over frames 0..<%d", minCos, stable))
        }
    }
    if let refPath = args.str("ref") {
        let ref = try loadArrays(url: URL(fileURLWithPath: refPath))
        let nr = min(frames, ref["latents"]!.dim(0))
        print("— vs upstream torch CPU bf16 (first \(nr) frames) —")
        var minCos = 1.0
        for f in 0..<nr {
            let (r, c) = compare(ref["latents"]![f], result.latents[0, f])
            minCos = min(minCos, c)
            print(String(format: "  latent frame %2d  rel %.4f  cos %.6f", f, r, c))
        }
        let n = min(nr * 4 - FF.droppedLeadingFrames, ref["pixels"]!.dim(0))
        let ps = psnr(ref["pixels"]![..<n], pixels[..<n])
        print(String(format: "  pixels (%d frames)  PSNR min %.2f dB  mean %.2f dB", n, ps.min()!,
                     ps.reduce(0, +) / Double(ps.count)))
        // V0 Python-MLX vs upstream: cos ≥ 0.9975, PSNR min 32.3 dB over the same span.
        // Floor 0.9965, was 0.997 (AB-R-1009, 2026-11-19): the torch oracle can't follow MLX kernel changes, and frame 2
        // already sits in the amplifying regime — mlx-swift 0.32.4 scored 0.99723, 0.33.1 scores 0.99694 with per-step
        // error unchanged (teacher-forced x0 rel 1.2–2.3e-6 per frame; Python oracle-vs-oracle rollout spread 7e-4).
        gate("rollout vs upstream", minCos >= 0.9965 && ps.min()! >= 30, String(format: "min cos %.5f", minCos))
    }
    if let out = args.str("out") {
        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try save(arrays: ["pixels": pixels, "latents": result.latents[0].asType(.float32)],
                 url: dir.appendingPathComponent("swift_rollout.safetensors"))
        let json = try JSONSerialization.data(withJSONObject: ["runs": report], options: [.prettyPrinted])
        try json.write(to: dir.appendingPathComponent("timing_swift.json"))
        try writeMP4(pixels, to: dir.appendingPathComponent("rollout.mp4"))
        print("wrote \(dir.path)/{rollout.mp4,timing_swift.json,swift_rollout.safetensors}")
    }

case "onestep":
    // Teacher-forced one-frame steps: the per-step error the rollout gate can't see past its own amplification.
    // For each frame k the KV cache is the oracle's (slots 0..<k final, as the oracle's rollout left them); Swift runs
    // frame k's denoise steps and context write; x0_k must match the oracle's at rel ≤ 5e-5. Measured 2026-11-19
    // (AB-R-1009): worst 1.08e-5 (frame 0, 4 steps + the clean pass), frames 1–13 at 1.2–2.3e-6 on mlx-swift 0.32.4 and
    // 0.33.1; Python mlx 0.31.2 vs 0.32.4 oracles score the same against each other (frame 0 6.2e-6).
    let tol = args.double("tol") ?? 5e-5
    let p = try loadPipeline(.float32)
    let inputs = try loadArrays(url: args.url("inputs"))
    let O = try loadArrays(url: args.url("oracle"))
    let nF = min(args.int("frames", 14), O["latents"]!.dim(0))
    let noise = TableNoise(arrays: inputs, dtype: .float32)
    let text = p.dit.textCache(inputs["prompt_embeds"]![.newAxis])
    let kv = FrameKVCache(frames: nF, dtype: .float32)
    let refE = p.refiner.embedTimestep(FF.refineTimestep)
    eval(text.arrays)
    func relMax(_ ref: MLXArray, _ got: MLXArray) -> (Float, Float) {
        let d = got.asType(.float32) - ref.asType(.float32)
        let rel = (d.square().sum().sqrt() / ref.asType(.float32).square().sum().sqrt()).item(Float.self)
        return (rel, abs(d).max().item(Float.self))
    }
    func slot(_ arrs: [MLXArray], _ f: Int) -> MLXArray {
        let (s, e) = (f * FF.frameTokens, (f + 1) * FF.frameTokens)
        return stacked(arrs.map { $0[0, 0..., s..<e, 0...].transposed(1, 0, 2).reshaped(FF.frameTokens, FF.dim) })
    }
    let oK = (0..<FF.layers).map { O["k\($0)"]![0..., 0..., ..<(nF * FF.frameTokens), 0...] }
    let oV = (0..<FF.layers).map { O["v\($0)"]![0..., 0..., ..<(nF * FF.frameTokens), 0...] }
    print("frame  x0 rel      x0 maxAbs   | slot K rel  slot V rel")
    var worst: Float = 0
    for frame in 0..<nF {
        kv.k = oK.map { $0 + 0 }  // fresh objects: slice assignment mutates an MLXArray in place
        kv.v = oV.map { $0 + 0 }
        let schedule = frame == 0 ? FF.firstTimesteps : FF.timesteps
        var x = noise.initial(frame: frame)
        var x0 = x
        for (step, t) in schedule.enumerated() {
            let (flow, _) = p.dit.forwardFrame(x, p.dit.timeConditioning(t), text, kv, frame: frame)
            x0 = p.scheduler.flowToX0(flow, x, t)
            if step + 1 < schedule.count {
                x = p.scheduler.addNoise(x0, noise.transition(frame: frame, step: step), schedule[step + 1])
            }
            eval([x0] + kv.arrays)
        }
        if frame == 0 {
            _ = p.dit.forwardFrame(x0, p.dit.timeConditioning(0.0), text, kv, frame: 0)
        } else {
            let phi = p.refiner.computePhi(p.dit.patchEmbed(x0), refE)
            let (k, v) = p.refiner.repairAll(slot(kv.k, frame), slot(kv.v, frame), phi, refE)
            let (s, e) = (frame * FF.frameTokens, (frame + 1) * FF.frameTokens)
            for i in 0..<FF.layers {
                kv.k[i][0..., 0..., s..<e, 0...] =
                    k[i].reshaped(FF.frameTokens, FF.heads, FF.headDim).transposed(1, 0, 2)[.newAxis]
                kv.v[i][0..., 0..., s..<e, 0...] =
                    v[i].reshaped(FF.frameTokens, FF.heads, FF.headDim).transposed(1, 0, 2)[.newAxis]
            }
        }
        eval(kv.arrays)
        let (xr, xm) = relMax(O["latents"]![frame][.newAxis], x0)
        let (kr, _) = relMax(slot(oK, frame), slot(kv.k, frame))
        let (vr, _) = relMax(slot(oV, frame), slot(kv.v, frame))
        worst = max(worst, xr)
        print(String(format: "%5d  %.3e  %.3e   | %.3e   %.3e", frame, xr, xm, kr, vr))
    }
    gate("one-step x0 vs Python-MLX (teacher-forced)", worst <= Float(tol),
         String(format: "worst rel %.2e over frames 0..<%d (tol %.0e)", worst, nF, tol))

case "t5":
    let t = now()
    let t5Dtype: DType = args.str("t5-dtype") == "f32" ? .float32 : .bfloat16
    let enc = try await FastFramesTextEncoder(
        wanDirectory: args.url("wan-dir"), tokenizerDirectory: args.url("tokenizer"), dtype: t5Dtype)
    let emb = enc.encode(args.req("prompt"))
    eval(emb)
    print(String(format: "umT5 (%@) load + encode %.2f s, MLX peak %@, footprint peak %@", "\(t5Dtype)", now() - t, gb(Memory.peakMemory), gb(peakFootprint())))
    let ref = try loadArrays(url: args.url("inputs"))["prompt_embeds"]!
    let nz = (ref.abs().sum(axis: -1) .> 0).sum().item(Int.self)
    let gotNz = (emb[0].abs().sum(axis: -1) .> 0).sum().item(Int.self)
    let (r, c) = compare(ref, emb[0])
    gate("umT5 vs upstream CPU bf16", gotNz == nz && c >= 0.999,
         String(format: "tokens %d/%d  rel %.4f  cos %.6f", gotNz, nz, r, c))

case "generate":
    let prompt = args.req("prompt")
    let seed = UInt64(args.int("seed", 0))
    let dir = URL(fileURLWithPath: args.req("out"))
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var t = now()
    var emb: MLXArray
    do {
        let enc = try await FastFramesTextEncoder(
            wanDirectory: args.url("wan-dir"), tokenizerDirectory: args.url("tokenizer"))
        emb = enc.encode(prompt)
        eval(emb)
    }
    Memory.clearCache()
    print(String(format: "prompt encoded in %.2f s (umT5 evicted)", now() - t))
    let p = try loadPipeline(.bfloat16)
    Memory.peakMemory = 0
    t = now()
    let r = p.rollout(promptEmbeds: emb, noise: SeededNoise(seed: seed), log: { print($0) })
    eval(r.latents, r.hidden, r.modulation)
    let rollS = now() - t
    t = now()
    let pixels = p.decoder.decodeClip(r.hidden, r.modulation)
    eval(pixels)
    print(String(format: "rollout %.2f s, decode %.2f s, MLX peak %@, footprint peak %@",
                 rollS, now() - t, gb(Memory.peakMemory), gb(peakFootprint())))
    try writeMP4(pixels, to: dir.appendingPathComponent("generate.mp4"))
    print("wrote \(dir.path)/generate.mp4")

case "engine":
    // --store <dir>: no explicit DiT — the engine materializes it from the Hub into that model store before load
    // (anonymous Hub access: an unsigned CLI touching a Keychain item blocks on an OS prompt).
    let store = args.str("store").map { URL(fileURLWithPath: $0) }
    let cfg = FastFramesConfiguration(
        ditFile: store == nil ? args.url("dit") : nil, addonsDirectory: args.url("addons"),
        textEncoderDirectory: args.url("wan-dir"), tokenizerDirectory: args.url("tokenizer"))
    let engine = store == nil ? MLXServeEngine() : MLXServeEngine(hfTokenProvider: { nil })
    if let store {
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        await engine.useModelStore(ModelStore(root: store))
        print("model store \(store.path): missing before run = \(cfg.missingWeightSources(storeRoot: store).map(\.repo))")
    }
    var t = now()
    let id = try await engine.register(FastFramesPackage.registration, configuration: cfg)
    print(String(format: "registered in %.3f s; licence advisories: %@", now() - t,
                 "\(await engine.licenseAdvisories.map(\.summary))"))
    let frames = args.int("frames", 81)
    let request = T2VRequest(prompt: args.req("prompt"), numFrames: frames,
                             seed: UInt64(args.int("seed", 0)))
    if args.str("warmup") != nil {
        t = now()
        _ = try await engine.run(T2VRequest(prompt: "warm-up", numFrames: 5, seed: 1), package: id)
        print(String(format: "warm-up (load + 5-frame run) %.2f s", now() - t))
    }
    Memory.clearCache()
    let floor = currentFootprint()
    t = now()
    let resp = try await engine.run(request, package: id)
    let runS = now() - t
    let peak = peakFootprint()
    guard let video = (resp as? T2VResponse)?.video else { fail("bad response") }
    print(String(format: "engine: %d frames in %.2f s (load %@); phys floor %@, lifetime peak %@ "
                 + "(activation ≈ %@); MLX peak %@; %d bytes mp4, %.2f s at %.0f fps",
                 frames, runS, args.str("warmup") == nil ? "included" : "excluded", gb(floor), gb(peak),
                 gb(peak - floor), gb(Memory.peakMemory), video.data.count, video.durationSeconds ?? 0,
                 video.frameRate ?? 0))
    if let out = args.str("out") {
        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try video.data.write(to: dir.appendingPathComponent("engine.mp4"))
        print("wrote \(dir.path)/engine.mp4")
    }

default:
    print("usage: ff-smoke lanes|rollout|onestep|t5|generate|engine [--options]  (see the header of main.swift)")
}

if !failures.isEmpty {
    print("FAILED: \(failures.joined(separator: ", "))")
    exit(1)
}
