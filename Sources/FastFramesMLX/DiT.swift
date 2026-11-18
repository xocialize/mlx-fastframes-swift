// Frame-causal Wan2.1-1.3B (`CausalWanModel.forward_frame`), the Causal Forcing++ framewise-2step generator.
// Weights keep upstream's torch key names (`blocks.N.self_attn.q.weight`, `patch_embedding.weight` …).

import Foundation
import MLX
import MLXFast
import MLXNN

/// Preallocated self-attention K/V for every layer, `[1, H, frames·1560, D]`, written in place one frame slot at a
/// time (the mlx-lm cache pattern). Fixed 21-frame window: no rolling past 81 RGB frames.
public final class FrameKVCache {
    public var k: [MLXArray]
    public var v: [MLXArray]
    public let frames: Int

    public init(frames: Int = FF.numFrames, dtype: DType) {
        self.frames = frames
        let shape = [1, FF.heads, frames * FF.frameTokens, FF.headDim]
        self.k = (0..<FF.layers).map { _ in MLXArray.zeros(shape, dtype: dtype) }
        self.v = (0..<FF.layers).map { _ in MLXArray.zeros(shape, dtype: dtype) }
    }

    public var arrays: [MLXArray] { k + v }

    /// One frame's K (or V) across layers as `[30, 1560, dim]` (the refiner's layout).
    func slot(_ frame: Int, keys: Bool) -> MLXArray {
        let (s, e) = (frame * FF.frameTokens, (frame + 1) * FF.frameTokens)
        let src = keys ? k : v
        return stacked(src.map {
            $0[0, 0..., s..<e, 0...].transposed(1, 0, 2).reshaped(FF.frameTokens, FF.dim)
        })
    }

    /// Overwrite one frame's slot from `[30, 1560, dim]`.
    func write(_ frame: Int, k newK: MLXArray, v newV: MLXArray) {
        let (s, e) = (frame * FF.frameTokens, (frame + 1) * FF.frameTokens)
        for i in 0..<FF.layers {
            k[i][0..., 0..., s..<e, 0...] =
                newK[i].reshaped(FF.frameTokens, FF.heads, FF.headDim).transposed(1, 0, 2)[.newAxis]
            v[i][0..., 0..., s..<e, 0...] =
                newV[i].reshaped(FF.frameTokens, FF.heads, FF.headDim).transposed(1, 0, 2)[.newAxis]
        }
    }
}

/// Cross-attention K/V per layer, computed once per prompt.
public struct TextCache {
    public let k: [MLXArray]
    public let v: [MLXArray]
    public var arrays: [MLXArray] { k + v }
}

/// Time conditioning for one timestep: block modulation input `e0 [1, 6, dim]` and head input `eHead [1, 1, dim]`.
public struct TimeConditioning {
    public let e0: MLXArray
    public let eHead: MLXArray
}

public final class CausalDiT {
    public let dtype: DType
    public let p: Params
    let patchW: MLXArray
    let patchB: MLXArray
    let rope: RopeTables
    let scale: Float

    public init(weights: [String: MLXArray], dtype: DType = .bfloat16) {
        self.dtype = dtype
        self.p = Params(weights, dtype: dtype)
        let w = p["patch_embedding.weight"]  // [dim, 16, 1, 2, 2]
        self.patchW = w.reshaped(FF.dim, -1).T  // (c, ph, pw) order
        self.patchB = p["patch_embedding.bias"]
        self.rope = RopeTables()
        self.scale = 1.0 / Float(FF.headDim).squareRoot()
    }

    /// `[B, 16, H, W]` → `[B, L, dim]`: Conv3d k = s = (1, 2, 2) as a GEMM.
    public func patchEmbed(_ latent: MLXArray) -> MLXArray {
        let (b, c, h, w) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        let x = latent.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 2, 4, 1, 3, 5)
        return matmul(x.reshaped(b, (h / 2) * (w / 2), c * 4), patchW) + patchB
    }

    public func timeConditioning(_ t: Double) -> TimeConditioning {
        var e = sinusoidal(FF.freqDim, t).asType(dtype)
        e = linear(silu(linear(e, p, "time_embedding.0")), p, "time_embedding.2")
        let e0 = linear(silu(e), p, "time_projection.1").reshaped(1, 6, FF.dim)
        return TimeConditioning(e0: e0, eHead: e.reshaped(1, 1, FF.dim))
    }

    /// `promptEmbeds [1, 512, 4096]` (zero-padded umT5 features) → per-layer cross-attention K/V.
    public func textCache(_ promptEmbeds: MLXArray) -> TextCache {
        var ctx = promptEmbeds.asType(dtype)
        ctx = linear(geluApproximate(linear(ctx, p, "text_embedding.0")), p, "text_embedding.2")
        var ks: [MLXArray] = []
        var vs: [MLXArray] = []
        for i in 0..<FF.layers {
            let pre = "blocks.\(i).cross_attn."
            let k = rmsNorm(linear(ctx, p, pre + "k"), p[pre + "norm_k.weight"])
            let v = linear(ctx, p, pre + "v")
            ks.append(k.reshaped(1, -1, FF.heads, FF.headDim).transposed(0, 2, 1, 3))
            vs.append(v.reshaped(1, -1, FF.heads, FF.headDim).transposed(0, 2, 1, 3))
        }
        return TextCache(k: ks, v: vs)
    }

    /// One latent frame through all 30 blocks, attending to the cached prefix. Writes this frame's K/V into
    /// `kv` and returns `(flow [1, 16, 60, 104], pre-head hidden states [1, 1560, dim])`.
    public func forwardFrame(
        _ latent: MLXArray, _ tc: TimeConditioning, _ text: TextCache, _ kv: FrameKVCache, frame: Int
    ) -> (flow: MLXArray, hidden: MLXArray) {
        var x = patchEmbed(latent)
        let cosF = rope.cos[frame]
        let sinF = rope.sin[frame]
        let (start, end) = (frame * FF.frameTokens, (frame + 1) * FF.frameTokens)
        let (h, d) = (FF.heads, FF.headDim)
        for i in 0..<FF.layers {
            let pre = "blocks.\(i)."
            let m = p[pre + "modulation"] + tc.e0  // [1, 6, dim]
            let e = (0..<6).map { m[0..., $0..<($0 + 1)] }
            var hs = layerNorm(x) * (1 + e[1]) + e[0]
            var q = rmsNorm(linear(hs, p, pre + "self_attn.q"), p[pre + "self_attn.norm_q.weight"])
            var k = rmsNorm(linear(hs, p, pre + "self_attn.k"), p[pre + "self_attn.norm_k.weight"])
            let v = linear(hs, p, pre + "self_attn.v").reshaped(1, -1, h, d)
            q = applyRope(q.reshaped(1, -1, h, d), cosF, sinF)
            k = applyRope(k.reshaped(1, -1, h, d), cosF, sinF)
            kv.k[i][0..., 0..., start..<end, 0...] = k.transposed(0, 2, 1, 3)
            kv.v[i][0..., 0..., start..<end, 0...] = v.transposed(0, 2, 1, 3)
            var y = MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3),
                keys: kv.k[i][0..., 0..., ..<end, 0...],
                values: kv.v[i][0..., 0..., ..<end, 0...],
                scale: scale, mask: nil)
            y = linear(y.transposed(0, 2, 1, 3).reshaped(1, -1, FF.dim), p, pre + "self_attn.o")
            x = x + y * e[2]
            let hc = layerNorm(x, p[pre + "norm3.weight"], p[pre + "norm3.bias"])
            let qc = rmsNorm(linear(hc, p, pre + "cross_attn.q"), p[pre + "cross_attn.norm_q.weight"])
            let yc = MLXFast.scaledDotProductAttention(
                queries: qc.reshaped(1, -1, h, d).transposed(0, 2, 1, 3),
                keys: text.k[i], values: text.v[i], scale: scale, mask: nil)
            x = x + linear(yc.transposed(0, 2, 1, 3).reshaped(1, -1, FF.dim), p, pre + "cross_attn.o")
            hs = layerNorm(x) * (1 + e[4]) + e[3]
            hs = linear(geluApproximate(linear(hs, p, pre + "ffn.0")), p, pre + "ffn.2")
            x = x + hs * e[5]
        }
        let hm = p["head.modulation"] + tc.eHead  // [1, 2, dim]
        let out = linear(
            layerNorm(x) * (1 + hm[0..., 1..<2]) + hm[0..., 0..<1], p, "head.head")  // [1, L, 64]
        let flow = out.reshaped(FF.gridH, FF.gridW, 2, 2, FF.latentChannels)
            .transposed(4, 0, 2, 1, 3)
            .reshaped(1, FF.latentChannels, FF.latentH, FF.latentW)
        return (flow, x)
    }
}
