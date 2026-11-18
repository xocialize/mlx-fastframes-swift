// The FastFrames autoregressive rollout (`FastFrames.rollout` + `decode`, stock path): per latent frame, a few
// FlowMatch steps against the cached prefix, then a context write — a clean t=0 pass for frame 0, the KV refiner
// for every later frame.

import Foundation
import MLX
import MLXRandom

/// Where the rollout's Gaussian noise comes from: the initial latent per frame and the re-noising draws between
/// steps (`transition(frame, step)` is used before step `step + 1`).
public protocol NoiseSource {
    func initial(frame: Int) -> MLXArray  // [1, 16, 60, 104]
    func transition(frame: Int, step: Int) -> MLXArray  // [1, 16, 60, 104]
}

/// Seeded MLX noise. Not bit-compatible with upstream's torch.Generator stream (gates use `TableNoise`).
public struct SeededNoise: NoiseSource {
    public let seed: UInt64
    public let dtype: DType
    public init(seed: UInt64, dtype: DType = .bfloat16) { self.seed = seed; self.dtype = dtype }

    func draw(_ index: Int) -> MLXArray {
        let key = MLXRandom.key(seed &+ UInt64(index) &* 1_000_003)
        return MLXRandom.normal([1, FF.latentChannels, FF.latentH, FF.latentW], key: key).asType(dtype)
    }
    public func initial(frame: Int) -> MLXArray { draw(frame) }
    public func transition(frame: Int, step: Int) -> MLXArray { draw(1000 + frame * 8 + step) }
}

/// Noise read from a table (the V0 spike's shared inputs: `noise [21, 16, 60, 104]`, `eps_<frame>_<step>`).
public struct TableNoise: NoiseSource {
    public let arrays: [String: MLXArray]
    public let dtype: DType
    public init(arrays: [String: MLXArray], dtype: DType = .bfloat16) { self.arrays = arrays; self.dtype = dtype }

    public func initial(frame: Int) -> MLXArray {
        guard let n = arrays["noise"] else { fatalError("TableNoise: no 'noise'") }
        return n[frame][.newAxis].asType(dtype)
    }
    public func transition(frame: Int, step: Int) -> MLXArray {
        guard let e = arrays["eps_\(frame)_\(step)"] else { fatalError("TableNoise: no eps_\(frame)_\(step)") }
        return e[.newAxis].asType(dtype)
    }
}

public struct FrameTiming: Codable {
    public let frame: Int
    public let denoiseSeconds: Double
    public let contextSeconds: Double
}

public struct RolloutResult {
    /// `[1, F, 16, 60, 104]`
    public let latents: MLXArray
    /// Pre-head hidden states `[1, F·1560, dim]` — the pixel decoder's input.
    public let hidden: MLXArray
    /// Head modulation input per frame `[1, F, 1, dim]`.
    public let modulation: MLXArray
    public let timings: [FrameTiming]
}

public final class FastFramesPipeline {
    public let dit: CausalDiT
    public let refiner: KVRefiner
    public let decoder: PixelDecoder
    public let scheduler = FlowMatch()
    private var tcCache: [Double: TimeConditioning] = [:]

    public init(dit: CausalDiT, refiner: KVRefiner, decoder: PixelDecoder) {
        self.dit = dit
        self.refiner = refiner
        self.decoder = decoder
    }

    /// Load the three weight files (upstream key names; the DiT is converted from Causal Forcing's
    /// `framewise-2step.pt` `generator_ema`).
    public static func load(dit: URL, refiner: URL, decoder: URL, dtype: DType = .bfloat16) throws -> FastFramesPipeline {
        let p = FastFramesPipeline(
            dit: CausalDiT(weights: try loadArrays(url: dit), dtype: dtype),
            refiner: KVRefiner(weights: try loadArrays(url: refiner), dtype: dtype),
            decoder: PixelDecoder(weights: try loadArrays(url: decoder)))
        eval(p.dit.p.all + p.refiner.p.all + p.decoder.p.all)
        return p
    }

    func tc(_ t: Double) -> TimeConditioning {
        if let c = tcCache[t] { return c }
        let c = dit.timeConditioning(t)
        tcCache[t] = c
        return c
    }

    /// Autoregressive rollout over `frames` latent frames (≤ 21). Causal: the first N frames of a full run equal an
    /// N-frame run with the same noise.
    /// `onFrame(frame)` runs after each latent frame is complete (cancellation / progress seam); a throw ends the
    /// rollout.
    public func rollout(
        promptEmbeds: MLXArray, noise: NoiseSource, frames: Int = FF.numFrames,
        log: ((String) -> Void)? = nil, onFrame: ((Int) throws -> Void)? = nil
    ) rethrows -> RolloutResult {
        precondition(frames >= 1 && frames <= FF.numFrames)
        let text = dit.textCache(promptEmbeds)
        // Sized to the clip: attention only ever reads the filled prefix, so a short clip needs no 21-frame cache.
        let kv = FrameKVCache(frames: frames, dtype: dit.dtype)
        eval(text.arrays + kv.arrays)
        let refE = refiner.embedTimestep(FF.refineTimestep)
        var latents: [MLXArray] = []
        var hidden: [MLXArray] = []
        var mods: [MLXArray] = []
        var timings: [FrameTiming] = []
        for frame in 0..<frames {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let schedule = frame == 0 ? FF.firstTimesteps : FF.timesteps
            var x = noise.initial(frame: frame)
            var x0 = x
            var feat = x
            var eh = x
            for (step, t) in schedule.enumerated() {
                let c = tc(t)
                let (flow, f) = dit.forwardFrame(x, c, text, kv, frame: frame)
                x0 = scheduler.flowToX0(flow, x, t)
                feat = f
                eh = c.eHead
                if step + 1 < schedule.count {
                    x = scheduler.addNoise(x0, noise.transition(frame: frame, step: step), schedule[step + 1])
                }
                eval([x0, feat] + kv.arrays)
            }
            latents.append(x0)
            hidden.append(feat)
            mods.append(eh)
            let t1 = DispatchTime.now().uptimeNanoseconds
            if frame == 0 {
                _ = dit.forwardFrame(x0, tc(0.0), text, kv, frame: 0)
            } else {
                let stem = dit.patchEmbed(x0)
                let phi = refiner.computePhi(stem, refE)
                let (k, v) = refiner.repairAll(kv.slot(frame, keys: true), kv.slot(frame, keys: false), phi, refE)
                kv.write(frame, k: k, v: v)
            }
            eval(kv.arrays)
            let t2 = DispatchTime.now().uptimeNanoseconds
            let timing = FrameTiming(
                frame: frame, denoiseSeconds: Double(t1 - t0) / 1e9, contextSeconds: Double(t2 - t1) / 1e9)
            timings.append(timing)
            log?(String(format: "frame %2d: denoise %.3f s, context %.3f s", frame, timing.denoiseSeconds,
                        timing.contextSeconds))
            try onFrame?(frame)
        }
        return RolloutResult(
            latents: stacked(latents, axis: 1), hidden: concatenated(hidden, axis: 1),
            modulation: stacked(mods, axis: 1), timings: timings)
    }
}
