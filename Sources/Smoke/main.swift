// ff-smoke — V2 gates and runs for the FastFrames-S Swift core.
//
//   lanes     fp32 component lanes vs upstream (torch CPU fp32, attention kept fp32): DiT flow / hidden / K cache,
//             refiner phi / K′ / V′, pixel decoder. Run with MLX_ENABLE_TF32=0.
//   rollout   bf16 rollout from the spike's shared noise table, compared per frame with the Python-MLX V0 rollout
//             (--pymlx; --dtype f32 with a pymlx_rollout_fp32.py oracle) and upstream's first 3 frames (--ref); times every frame; optional --out writes an MP4,
//             timing.json and the uint8 pixels.
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
        //   fp32: max latent rel over frames 0..<stable (default 14) ≤ 1e-3 — the logic gate
        //   bf16: min latent cos over frames 0..<stable (default 3) ≥ 0.997 — the backend-rounding band V0 measured
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
            gate("rollout vs Python-MLX (fp32 logic)", maxRel <= 1e-3,
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
        gate("rollout vs upstream", minCos >= 0.997 && ps.min()! >= 30, String(format: "min cos %.5f", minCos))
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
    print("usage: ff-smoke lanes|rollout|t5|generate|engine [--options]  (see the header of main.swift)")
}

if !failures.isEmpty {
    print("FAILED: \(failures.joined(separator: ", "))")
    exit(1)
}
