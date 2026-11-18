// Shared constants, the FlowMatch schedule, Wan's 3-D RoPE tables and the small ops every module uses.
// Line-for-line with the V0 Python-MLX spike (mlxengine-video/WIP/fastframes-eval/spike/ff_mlx.py), which is
// gated against upstream apple/ml-fastframes @ a65c592 (stock pipeline.py path, fast_ops=False).

import Foundation
import MLX
import MLXFast
import MLXNN

/// FF-S geometry: Wan2.1-1.3B at 832×480, 21 latent frames (81 RGB frames).
public enum FF {
    public static let dim = 1536
    public static let ffnDim = 8960
    public static let heads = 12
    public static let layers = 30
    public static let headDim = dim / heads  // 128
    public static let freqDim = 256
    public static let textLen = 512
    public static let textDim = 4096
    public static let latentChannels = 16
    public static let numFrames = 21
    public static let latentH = 60
    public static let latentW = 104
    public static let gridH = latentH / 2  // 30
    public static let gridW = latentW / 2  // 52
    public static let frameTokens = gridH * gridW  // 1560
    /// Frame 0: four denoising steps, then a clean t=0 pass that writes its K/V.
    public static let firstTimesteps: [Double] = [1000.0, 937.5, 833.3333129882812, 625.0]
    /// Frames 1…20: two steps, then the KV refiner rewrites the frame's cache slot.
    public static let timesteps: [Double] = [1000.0, 833.3333129882812]
    public static let refineTimestep: Double = 833.3333129882812
    /// The decoder emits 4 RGB frames per latent frame (84) and the first 3 are dropped (81).
    public static let droppedLeadingFrames = 3
    public static let width = gridW * 16  // 832
    public static let height = gridH * 16  // 480
}

/// Flat safetensors parameter map with checked lookups.
public struct Params {
    public private(set) var arrays: [String: MLXArray]

    public init(_ arrays: [String: MLXArray], dtype: DType) {
        self.arrays = arrays.mapValues { $0.dtype == dtype ? $0 : $0.asType(dtype) }
    }

    public subscript(_ key: String) -> MLXArray {
        guard let a = arrays[key] else { fatalError("FastFrames: missing weight '\(key)'") }
        return a
    }

    public func optional(_ key: String) -> MLXArray? { arrays[key] }

    public var all: [MLXArray] { Array(arrays.values) }
}

/// `FlowMatchScheduler(shift=5)`: 1000-step grid, nearest-timestep sigma lookup (numpy float32 arithmetic).
public struct FlowMatch {
    public let sigmas: [Float]
    public let timesteps: [Float]

    public init(shift: Float = 5.0, steps: Int = 1000) {
        // np.linspace(1, 0, steps + 1, dtype=float32)[:-1]: float64 ramp, cast to float32
        let step = -1.0 / Double(steps)
        var sig = [Float](repeating: 0, count: steps)
        for i in 0..<steps {
            let s = Float(Double(i) * step + 1.0)
            sig[i] = shift * s / (1 + (shift - 1) * s)
        }
        self.sigmas = sig
        self.timesteps = sig.map { $0 * 1000 }
    }

    public func sigma(_ t: Double) -> Float {
        let tf = Float(t)
        var best = 0
        var bestD = Float.infinity
        for (i, ts) in timesteps.enumerated() {
            let d = abs(ts - tf)
            if d < bestD { bestD = d; best = i }
        }
        return sigmas[best]
    }

    /// x0 = xt − σ·flow (upstream: float64; MLX GPU: float32).
    public func flowToX0(_ flow: MLXArray, _ xt: MLXArray, _ t: Double) -> MLXArray {
        let s = sigma(t)
        return (xt.asType(.float32) - s * flow.asType(.float32)).asType(flow.dtype)
    }

    /// xt = (1 − σ)·x0 + σ·ε.
    public func addNoise(_ x0: MLXArray, _ eps: MLXArray, _ t: Double) -> MLXArray {
        let s = sigma(t)
        let oneMinus = Float(1.0 - Double(s))
        return (oneMinus * x0.asType(.float32) + s * eps.asType(.float32)).asType(eps.dtype)
    }
}

/// `sinusoidal_embedding_1d` (cos, sin) computed in float64, returned as float32 `[1, dim]`.
func sinusoidal(_ dim: Int, _ t: Double) -> MLXArray {
    let half = dim / 2
    var out = [Float](repeating: 0, count: dim)
    for j in 0..<half {
        let a = t * pow(10000.0, -Double(j) / Double(half))
        out[j] = Float(cos(a))
        out[half + j] = Float(sin(a))
    }
    return MLXArray(out, [1, dim])
}

/// Per-frame RoPE tables `[F, L, 1, 64]` for Wan's 3-D RoPE (22 temporal / 21 height / 21 width pairs).
struct RopeTables {
    let cos: MLXArray
    let sin: MLXArray

    init(frames: Int = FF.numFrames) {
        let d = FF.headDim
        func freqs(_ dim: Int) -> [Double] {
            stride(from: 0, to: dim, by: 2).map { 1.0 / pow(10000.0, Double($0) / Double(dim)) }
        }
        let tF = freqs(d - 4 * (d / 6))  // 22
        let hF = freqs(2 * (d / 6))  // 21
        let wF = freqs(2 * (d / 6))  // 21
        let pairs = tF.count + hF.count + wF.count  // 64
        let n = frames * FF.frameTokens * pairs
        var c = [Float](repeating: 0, count: n)
        var s = [Float](repeating: 0, count: n)
        var i = 0
        for f in 0..<frames {
            for h in 0..<FF.gridH {
                for w in 0..<FF.gridW {
                    for (pos, table) in [(f, tF), (h, hF), (w, wF)] {
                        for fr in table {
                            let a = Double(pos) * fr
                            c[i] = Float(Foundation.cos(a))
                            s[i] = Float(Foundation.sin(a))
                            i += 1
                        }
                    }
                }
            }
        }
        self.cos = MLXArray(c, [frames, FF.frameTokens, 1, pairs])
        self.sin = MLXArray(s, [frames, FF.frameTokens, 1, pairs])
    }
}

/// Interleaved-pair rotation on `[B, L, H, D]` in float32 (upstream: complex128), cast back to the input dtype.
func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
    let shape = x.shape
    let p = x.asType(.float32).reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
    let re = p[.ellipsis, 0]
    let im = p[.ellipsis, 1]
    let out = stacked([re * cos - im * sin, re * sin + im * cos], axis: -1)
    return out.reshaped(shape).asType(x.dtype)
}

@inline(__always)
func layerNorm(_ x: MLXArray, _ w: MLXArray? = nil, _ b: MLXArray? = nil, eps: Float = 1e-6) -> MLXArray {
    MLXFast.layerNorm(x, weight: w, bias: b, eps: eps)
}

/// `WanRMSNorm`: normalize in float32, cast back, then scale.
@inline(__always)
func rmsNorm(_ x: MLXArray, _ w: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let n = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)
    return n.asType(x.dtype) * w
}

@inline(__always)
func linear(_ x: MLXArray, _ p: Params, _ name: String) -> MLXArray {
    let y = matmul(x, p[name + ".weight"].T)
    if let b = p.optional(name + ".bias") { return y + b }
    return y
}
