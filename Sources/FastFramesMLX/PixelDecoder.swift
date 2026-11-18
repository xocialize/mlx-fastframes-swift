// Causal ConvNeXt pixel decoder (Apple, AMLR): reads the DiT's pre-head hidden states plus the head modulation and
// emits RGB directly — no VAE for text-to-video. Channels-last `[B, F, h, w, C]`, float32 throughout. The depthwise
// 3×3×3 conv (two frames of left time padding) is written as 27 shifted multiply-adds, which sidesteps MLX's conv
// kernels and the lossy-Winograd question (AB-A-0115) entirely.

import Foundation
import MLX
import MLXFast
import MLXNN

public final class PixelDecoder {
    public let p: Params
    let blocks = ["conv_low.0.", "conv_low.1.", "conv_high.0."]

    public init(weights: [String: MLXArray]) {
        var stripped: [String: MLXArray] = [:]
        for (k, v) in weights {
            stripped[k.hasPrefix("pixel_head.") ? String(k.dropFirst("pixel_head.".count)) : k] = v
        }
        self.p = Params(stripped, dtype: .float32)
    }

    func pw(_ x: MLXArray, _ name: String) -> MLXArray {
        let w = p[name + ".weight"]
        return matmul(x, w.reshaped(w.dim(0), -1).T) + p[name + ".bias"]
    }

    func dwconv(_ x: MLXArray, _ name: String) -> MLXArray {
        let w = p[name + ".weight"].reshaped(-1, 3, 3, 3)  // [C, kt, kh, kw]
        let (f, h, wd) = (x.dim(1), x.dim(2), x.dim(3))
        let xp = padded(x, widths: [0, [2, 0], [1, 1], [1, 1], 0])
        var out = MLXArray.zeros(like: x) + p[name + ".bias"]
        for dt in 0..<3 {
            for dh in 0..<3 {
                for dw in 0..<3 {
                    out = out
                        + xp[0..., dt..<(dt + f), dh..<(dh + h), dw..<(dw + wd), 0...] * w[0..., dt, dh, dw]
                }
            }
        }
        return out
    }

    /// `hidden [1, F·1560, dim]`, `eMod [1, F, 1, dim]` → pixels in [-1, 1], `[1, 4F, 3, 480, 832]` (leading frames
    /// not yet dropped).
    public func callAsFunction(_ hidden: MLXArray, _ eMod: MLXArray) -> MLXArray {
        let nf = eMod.dim(1)
        var x = hidden.asType(.float32)
        let m = p["modulation"][0..., .newAxis] + eMod.asType(.float32)  // [1, F, 2, dim]
        let shift = m[0..., 0..., 0..<1]
        let scale = m[0..., 0..., 1..<2]
        x = layerNorm(x).reshaped(1, nf, FF.frameTokens, FF.dim) * (1 + scale) + shift
        var v = x.reshaped(1, nf, FF.gridH, FF.gridW, FF.dim)
        v = pw(layerNorm(v, p["entry_norm.weight"], p["entry_norm.bias"]), "entry_pw")
        for blk in blocks {
            var y = dwconv(v, blk + "dw")
            y = layerNorm(y, p[blk + "norm.weight"], p[blk + "norm.bias"])
            y = pw(gelu(pw(y, blk + "pw1")), blk + "pw2")
            v = v + y
        }
        let tok = matmul(v, p["fc_out.weight"].T) + p["fc_out.bias"]  // [1, F, h, w, 3072]
        let px = tok.reshaped(nf, FF.gridH, FF.gridW, 4, 16, 16, 3)
            .transposed(0, 3, 6, 1, 4, 2, 5)
            .reshaped(nf * 4, 3, FF.height, FF.width)
        return px[.newAxis]
    }

    /// Whole clip → `[81, 480, 832, 3]` uint8 (drops the decoder's first 3 frames).
    public func decodeClip(_ hidden: MLXArray, _ eMod: MLXArray) -> MLXArray {
        let px = self(hidden, eMod)[0, FF.droppedLeadingFrames...]
        return ((clip(px, min: -1, max: 1) + 1) * 127.5).round().asType(.uint8).transposed(0, 2, 3, 1)
    }
}
