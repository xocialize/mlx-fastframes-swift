import FastFramesMLX
import Foundation
import MLX
import MLXToolKit

/// Licences this package declares that the engine has no named constant for (yet). Kept in a namespace so a later
/// engine constant of the same meaning can't collide with them.
public enum FastFramesLicense {
    /// Apple Machine Learning Research Model License: research use only, non-commercial, revocable; binds
    /// derivatives. On the FastFrames-S KV refiner and pixel decoder weights.
    public static let amlr: SPDXLicense = "LicenseRef-Apple-AMLR"
    /// Apple Sample Code License: use, modify and redistribute; the notice is required only on unmodified
    /// redistribution; no use of Apple's name. On upstream's inference code, which this package translates.
    public static let appleSampleCode: SPDXLicense = "LicenseRef-Apple-Sample-Code"

    static let amlrTerms = "https://huggingface.co/apple/FastFrames-S/blob/main/LICENSE"
    static let codeTerms = "https://github.com/apple-aiml-research/ml-fastframes/blob/main/LICENSE"
}

public enum FastFramesPackageError: Error, CustomStringConvertible {
    case weightsNotProvisioned(missing: String)

    public var description: String {
        switch self {
        case .weightsNotProvisioned(let m):
            return "FastFrames-S weights not found (\(m)). The engine downloads the add-ons from apple/FastFrames-S and "
                + "the text encoder on first use; the converted DiT (\(FastFramesConfiguration.ditRepo)) is not "
                + "published yet — set FastFramesConfiguration.ditFile to a local dit.safetensors."
        }
    }
}

/// An MLXEngine `textToVideo` package over **Apple FastFrames-S** (arXiv 2611.23226): autoregressive two-step video
/// diffusion. A frame-causal Wan2.1-1.3B DiT (Causal Forcing++ framewise-2step) denoises one latent frame at a time
/// against a KV cache of the frames before it; Apple's KV refiner rewrites each finished frame's cache slot; Apple's
/// causal ConvNeXt pixel decoder turns the DiT's hidden states straight into RGB (no VAE).
///
/// A **fast draft** lane: 81 frames of 832×480 in ~11 s on an M5 Max (≈7 fps, ~0.34× real time at 16 fps). The
/// schedule is distilled (2 steps per frame, 4 + a clean pass for the first) with no classifier-free guidance.
///
/// **Research only.** The refiner and decoder weights are under the Apple Machine Learning Research Model License
/// (non-commercial, revocable). Admitted for opt-in, named-only Gaea plugins (AB-D-0259 under AB-D-0226 channel 2);
/// never a shipping app. A `.blocking` permissive-only host refuses it; the default advisory host registers it with
/// the finding recorded.
///
/// Lifecycle (C13): `load()` pages in the DiT, refiner and decoder (~3.8 GB). umT5-XXL (~11 GB bf16) is loaded per
/// request and evicted before denoising. Cancellation is checked after every latent frame.
@InferenceActor
public final class FastFramesPackage: ModelPackage {
    public typealias Configuration = FastFramesConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: primary checkpoint = the Causal Forcing++ DiT (Apache-2.0, HF card); Apple's add-ons are AMLR.
            // C8: the code translates upstream's inference code (Apple Sample Code License) — declared as such, with
            // the notice a linking host reads.
            license: LicenseDeclaration(
                weightLicense: .apache2,
                additionalWeightLicenses: [FastFramesLicense.amlr],
                portCodeLicense: FastFramesLicense.appleSampleCode,
                notices: [
                    LicenseNotice(
                        text: "Research use only, non-commercial. FastFrames-S's KV refiner and pixel decoder are an "
                            + "Apple Machine Learning Research Model: Apple Machine Learning Research Model is licensed "
                            + "under the Apple Machine Learning Research Model License Agreement. Apple may terminate "
                            + "the licence; on termination every copy must be deleted.",
                        display: .informational, license: FastFramesLicense.amlr,
                        termsURL: FastFramesLicense.amlrTerms),
                    LicenseNotice(
                        text: "This package's code is a translation of Apple's FastFrames inference code, used under "
                            + "the Apple Sample Code License (modify and redistribute; Apple's name and marks may not be "
                            + "used to endorse derived products).",
                        display: .informational, license: FastFramesLicense.appleSampleCode,
                        termsURL: FastFramesLicense.codeTerms),
                ]),
            provenance: Provenance(sourceRepo: FastFramesConfiguration.addonsRepo, revision: "main", tier: 2),
            requirements: RequirementsManifest(
                footprints: FastFramesPackage.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [SpecialtyWeight(.general, strength: 0.3)],
            surfaces: [
                T2VContract.descriptor(
                    name: "fastframes-s-t2v",
                    summary: "Apple FastFrames-S fast-draft text-to-video (MLX; research use only). 832x480 at 16 fps, "
                        + "up to 81 frames in ~11 s (shorter clips are proportionally faster). Distilled two-step "
                        + "schedule with no CFG: `steps`, `guidanceScale` and `negativePrompt` are refused, as are "
                        + "image conditioning and other sizes.")
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: FastFramesPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        if let missing = configuration.firstMissingFile() {
            throw FastFramesPackageError.weightsNotProvisioned(missing: missing)
        }
        let p = configuration.resolvedPaths()!
        pipeline = try FastFramesPipeline.load(dit: p.dit, refiner: p.refiner, decoder: p.decoder)
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()  // CAN-1: first act
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .textToVideo, let req = request as? T2VRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req)

        RunProgress.report(.encode)
        let paths = configuration.resolvedPaths()!
        let embeds: MLXArray
        do {
            let encoder = try await FastFramesTextEncoder(
                wanDirectory: paths.textEncoderDirectory, tokenizerDirectory: paths.tokenizerDirectory)
            embeds = encoder.encode(req.prompt)
            eval(embeds)
        }
        MLX.Memory.clearCache()  // umT5 is out of scope: hand its ~11 GB back before the rollout
        try Task.checkCancellation()

        let result = try pipeline.rollout(
            promptEmbeds: embeds, noise: SeededNoise(seed: plan.seed), frames: plan.latentFrames,
            onFrame: { frame in
                RunProgress.report(.denoise, step: frame + 1, totalSteps: plan.latentFrames)
                try Task.checkCancellation()
            })
        RunProgress.report(.decode)
        var pixels = pipeline.decoder.decodeClip(result.hidden, result.modulation)
        if pixels.dim(0) > plan.frames { pixels = pixels[..<plan.frames] }
        eval(pixels)
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        let mp4 = try encodeMP4(pixels, fps: plan.fps)
        return T2VResponse(video: Video(format: .mp4, data: mp4,
                                        durationSeconds: Double(plan.frames) / plan.fps, frameRate: plan.fps))
    }

    /// What a request asks of the fixed FastFrames-S geometry: RGB frames, latent frames to roll out, playback rate,
    /// seed. Refuses legibly what the model cannot honour.
    struct Plan: Equatable {
        let frames: Int
        let latentFrames: Int
        let fps: Double
        let seed: UInt64
    }

    nonisolated static func plan(_ req: T2VRequest) throws -> Plan {
        func refuse(_ why: String) -> PackageError { .unsupportedRequestFeature("FastFrames-S: " + why) }
        if req.initImage != nil { throw refuse("initImage (text-to-video only)") }
        if let refs = req.referenceImages, !refs.isEmpty { throw refuse("referenceImages") }
        if req.initAudio != nil { throw refuse("initAudio") }
        if !req.keyframes.isEmpty { throw refuse("keyframes") }
        if req.includeAudio == true { throw refuse("includeAudio (the model is silent)") }
        if !req.adapters.isEmpty { throw refuse("adapters") }
        if let n = req.negativePrompt, !n.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw refuse("negativePrompt (no classifier-free guidance)")
        }
        if let g = req.guidanceScale, g != 1 { throw refuse("guidanceScale (no classifier-free guidance)") }
        if let s = req.steps, s != FF.timesteps.count { throw refuse("steps (fixed distilled schedule: 2 per frame)") }
        if let w = req.width, w != FF.width { throw refuse("width \(w) (only \(FF.width))") }
        if let h = req.height, h != FF.height { throw refuse("height \(h) (only \(FF.height))") }
        let maxFrames = FF.numFrames * 4 - FF.droppedLeadingFrames  // 81
        let frames = req.numFrames ?? maxFrames
        guard frames >= 1, frames <= maxFrames else { throw refuse("numFrames \(frames) (1…\(maxFrames))") }
        let fps = req.fps ?? 16
        guard fps > 0 else { throw refuse("fps \(fps)") }
        // latent frame i yields RGB frames 4i…4i+3 before the first 3 are dropped
        let latent = (frames + FF.droppedLeadingFrames + 3) / 4
        return Plan(frames: frames, latentFrames: latent, fps: fps, seed: req.seed ?? UInt64.random(in: 0...UInt64(UInt32.max)))
    }
}

extension FastFramesPackage {
    public nonisolated static var registration: PackageRegistration { .of(FastFramesPackage.self) }

    /// Split footprint — MEASURED through the real MLXServeEngine (`ff-smoke engine`, 2026-11-18, M5 Max 128 GB,
    /// macOS 27.3, release build, mlx-swift 0.32.4; basis = process lifetime-peak `phys_footprint`; one measured run
    /// per process after a 5-frame warm-up):
    ///
    ///   run                               phys floor (post-load)   lifetime peak   MLX peak
    ///   81 frames, warm (load excluded)   6.86 GB                  16.08 GB        15.73 GB
    ///   33 frames, cold (load included)   —                        16.03 GB        15.73 GB
    ///
    /// The peak is the PROMPT phase, not the rollout: umT5-XXL (11.4 GB bf16) is paged in per request next to the
    /// resident set, then evicted before denoising (the rollout itself peaks ~12.5 GB: weights + the 6 GB KV cache).
    /// So the activation is flat in clip length. Resident = the weights as loaded: DiT 2.84 GB bf16 + refiner 0.69 GB
    /// bf16 + decoder 0.26 GB fp32 + RoPE tables ≈ 4.0 GB. Activation = (16.08 − 3.9) × 1.2 ≈ 14.7 GB.
    nonisolated static var footprints: [QuantFootprint] {
        [QuantFootprint(quant: .bf16, residentBytes: 4_000_000_000, peakActivationBytes: 14_700_000_000)]
    }
}
