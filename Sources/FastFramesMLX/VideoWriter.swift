// H.264 MP4 encoding of decoded clips (AVFoundation; no FFmpeg).

import AVFoundation
import CoreVideo
import Foundation
import MLX

/// H.264 MP4 from uint8 `[N, H, W, 3]` RGB frames.
public func writeMP4(_ frames: MLXArray, to url: URL, fps: Double = 16) throws {
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
        adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f * 1000), timescale: CMTimeScale((fps * 1000).rounded())))
    }
    input.markAsFinished()
    let done = DispatchSemaphore(value: 0)
    writer.finishWriting { done.signal() }
    done.wait()
    if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
}

/// The clip as MP4 bytes (written through a temporary file).
public func encodeMP4(_ frames: MLXArray, fps: Double) throws -> Data {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
    defer { try? FileManager.default.removeItem(at: url) }
    try writeMP4(frames, to: url, fps: fps)
    return try Data(contentsOf: url)
}
