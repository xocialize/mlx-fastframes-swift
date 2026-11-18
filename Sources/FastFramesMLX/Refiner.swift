// `KVRefinementNetwork` (Apple, AMLR): after a frame is denoised, predicts deltas on that frame's exact K/V for all
// 30 layers at once (grouped heads) and rewrites the cache slot — K renormalised per head to the original norm,
// V scaled by exp(s_V). d_a 768 for FF-S, 4 stem blocks over the clean latent's patch tokens.

import Foundation
import MLX
import MLXFast
import MLXNN

public final class KVRefiner {
    public let dtype: DType
    public let p: Params
    public let dA: Int

    public init(weights: [String: MLXArray], dtype: DType = .bfloat16) {
        self.dtype = dtype
        self.p = Params(weights, dtype: dtype)
        self.dA = p["stem_in.weight"].dim(0)
    }

    /// `[1, d_a]`.
    public func embedTimestep(_ t: Double) -> MLXArray {
        let e = sinusoidal(FF.freqDim, t).asType(dtype)
        return linear(silu(linear(e, p, "t_embed.mlp.0")), p, "t_embed.mlp.2")
    }

    /// Stem over the clean latent's patch tokens `[1, 1560, dim]` → `phi [1, 1560, d_a]`.
    public func computePhi(_ stemTokens: MLXArray, _ e: MLXArray) -> MLXArray {
        let d = dA
        var x = linear(stemTokens, p, "stem_in")
        for i in 0..<4 {
            let pre = "stem_blocks.\(i)."
            let mod = linear(silu(e), p, pre + "mod.lin")[0..., .newAxis, 0...]
            let c = (0..<6).map { mod[.ellipsis, ($0 * d)..<(($0 + 1) * d)] }
            let (sh1, sc1, g1, sh2, sc2, g2) = (c[0], c[1], c[2], c[3], c[4], c[5])
            var h = layerNorm(x, eps: 1e-5) * (1 + sc1) + sh1
            let qkv = linear(h, p, pre + "qkv").reshaped(1, -1, 3, 8, d / 8).transposed(2, 0, 3, 1, 4)
            let y = MLXFast.scaledDotProductAttention(
                queries: qkv[0], keys: qkv[1], values: qkv[2],
                scale: 1.0 / Float(d / 8).squareRoot(), mask: nil)
            x = x + g1 * linear(y.transposed(0, 2, 1, 3).reshaped(1, -1, d), p, pre + "proj")
            h = layerNorm(x, eps: 1e-5) * (1 + sc2) + sh2
            x = x + g2 * linear(geluApproximate(linear(h, p, pre + "mlp.0")), p, pre + "mlp.2")
        }
        return layerNorm(x, p["stem_norm.weight"], p["stem_norm.bias"], eps: 1e-5)
    }

    /// `nK, nV [30, S, dim]`; `phi [1, S, d_a]`; `e [1, d_a]` → refined `(K′, V′)`, same shapes.
    public func repairAll(_ nK: MLXArray, _ nV: MLXArray, _ phi: MLXArray, _ e: MLXArray) -> (MLXArray, MLXArray) {
        let d = dA
        let L = FF.layers
        let g = "heads_grouped.gemms.0."
        var phiV = matmul(phi[0], p["heads_grouped.phi_proj.\(d).weight"].T)[.newAxis]
        phiV = broadcast(phiV, to: [L, phiV.dim(1), phiV.dim(2)])
        let pk = matmul(nK, p[g + "proj_k_w"]) + p[g + "proj_k_b"]
        let pv = matmul(nV, p[g + "proj_v_w"]) + p[g + "proj_v_b"]
        var h = matmul(concatenated([pk, pv, phiV], axis: -1), p[g + "fuse_w"]) + p[g + "fuse_b"]
        let e1 = broadcast(silu(e).reshaped(1, 1, d), to: [L, 1, d])
        let mod = matmul(e1, p[g + "mod_w"]) + p[g + "mod_b"]
        let shift = mod[.ellipsis, ..<d]
        let scale = mod[.ellipsis, d...]
        h = layerNorm(h, eps: 1e-5) * (1 + scale) + shift
        h = matmul(geluApproximate(matmul(h, p[g + "m0_w"]) + p[g + "m0_b"]), p[g + "m2_w"]) + p[g + "m2_b"]
        let k = nK + (matmul(h, p[g + "dK_w"]) + p[g + "dK_b"])
        var kh = k.reshaped(L, -1, FF.heads, FF.headDim).asType(.float32)
        let rh = nK.reshaped(L, -1, FF.heads, FF.headDim).asType(.float32)
        let rn = sqrt((rh * rh).sum(axis: -1, keepDims: true))
        let kn = sqrt((kh * kh).sum(axis: -1, keepDims: true))
        kh = kh * (rn / (kn + 1e-6))
        let kOut = kh.reshaped(L, -1, FF.dim).asType(nK.dtype)
        let vOut = (nV + (matmul(h, p[g + "dV_w"]) + p[g + "dV_b"])) * exp(p[g + "v_logscale"])
        return (kOut, vOut)
    }
}
