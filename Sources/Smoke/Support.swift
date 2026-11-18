import AVFoundation
import CoreVideo
import Darwin
import Foundation
import MLX

struct Args {
    let mode: String
    private var kv: [String: String] = [:]

    init(_ argv: [String]) {
        mode = argv.count > 1 ? argv[1] : "help"
        var i = 2
        while i < argv.count {
            let k = argv[i]
            if k.hasPrefix("--") {
                if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                    kv[String(k.dropFirst(2))] = argv[i + 1]
                    i += 2
                } else {
                    kv[String(k.dropFirst(2))] = "1"
                    i += 1
                }
            } else {
                i += 1
            }
        }
    }

    func str(_ k: String) -> String? { kv[k] }
    func req(_ k: String) -> String {
        guard let v = kv[k] else { fail("missing --\(k)") }
        return v
    }
    func url(_ k: String) -> URL { URL(fileURLWithPath: req(k)) }
    func int(_ k: String, _ d: Int) -> Int { kv[k].flatMap(Int.init) ?? d }
    func double(_ k: String) -> Double? { kv[k].flatMap(Double.init) }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("ff-smoke: " + msg + "\n").data(using: .utf8)!)
    exit(2)
}

/// Relative L2 error and cosine in float64 on the host.
func compare(_ ref: MLXArray, _ got: MLXArray) -> (rel: Double, cos: Double) {
    let a = ref.asType(.float32).flattened().asArray(Float.self)
    let b = got.asType(.float32).flattened().asArray(Float.self)
    precondition(a.count == b.count, "size mismatch \(a.count) vs \(b.count)")
    var d2 = 0.0, a2 = 0.0, b2 = 0.0, ab = 0.0
    for i in 0..<a.count {
        let x = Double(a[i]), y = Double(b[i])
        d2 += (x - y) * (x - y); a2 += x * x; b2 += y * y; ab += x * y
    }
    return ((d2 / a2).squareRoot(), ab / (a2.squareRoot() * b2.squareRoot()))
}

/// Per-frame PSNR over uint8 `[N, H, W, 3]`.
func psnr(_ ref: MLXArray, _ got: MLXArray) -> [Double] {
    (0..<ref.dim(0)).map { i in
        let d = ref[i].asType(.float32) - got[i].asType(.float32)
        let mse = Double((d * d).mean().item(Float.self))
        return 10 * log10(255 * 255 / max(mse, 1e-12))
    }
}

/// Peak `phys_footprint` of this process so far (bytes) — the real working-set peak.
func peakFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak) : -1
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

/// H.264 MP4 from uint8 `[N, H, W, 3]` RGB frames.
func writeMP4(_ frames: MLXArray, to url: URL, fps: Int32 = 16) throws {
    try? FileManager.default.removeItem(at: url)
    let (n, h, w) = (frames.dim(0), frames.dim(1), frames.dim(2))
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
    ])
    writer.add(input)
    guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
    writer.startSession(atSourceTime: .zero)
    let rgb = frames.asArray(UInt8.self)
    for f in 0..<n {
        while !input.isReadyForMoreMediaData { usleep(1000) }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
        guard let buf = pb else { throw CocoaError(.fileWriteUnknown) }
        CVPixelBufferLockBaseAddress(buf, [])
        let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buf)
        rgb.withUnsafeBufferPointer { src in
            for y in 0..<h {
                for x in 0..<w {
                    let s = ((f * h + y) * w + x) * 3
                    let d = y * stride + x * 4
                    base[d] = src[s + 2]; base[d + 1] = src[s + 1]; base[d + 2] = src[s]; base[d + 3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buf, [])
        adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: fps))
    }
    input.markAsFinished()
    let done = DispatchSemaphore(value: 0)
    writer.finishWriting { done.signal() }
    done.wait()
    if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
}
