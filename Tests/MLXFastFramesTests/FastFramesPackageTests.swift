// FastFramesPackageTests — the package gates, offline: MAT (four sources, explicit paths win), the licence posture
// (AMLR declared and noticed; a `.blocking` permissive-only host refuses it; the default advisory host registers it
// with the findings recorded), FIT, CAN-1/CAN-3, request planning (frames → latent frames, legible refusals), the
// honest load refusal, and the configuration's host-path-free Codable form.

import FastFramesMLX
import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXFastFrames

final class FastFramesPackageTests: XCTestCase {

    static func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// A complete set of explicit component paths (placeholder bytes).
    static func satisfied() throws -> FastFramesConfiguration {
        let root = try tempDir()
        func touch(_ rel: String) throws -> URL {
            let u = root.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([0]).write(to: u)
            return u
        }
        let dit = try touch("dit/dit.safetensors")
        _ = try touch("addons/fastframes-s/kv_refiner.safetensors")
        _ = try touch("addons/fastframes-s/pixel_decoder.safetensors")
        _ = try touch("te/t5_encoder.safetensors")
        _ = try touch("te/config.json")
        for f in FastFramesConfiguration.tokenizerFiles { _ = try touch("tok/\(f)") }
        return FastFramesConfiguration(
            ditFile: dit, addonsDirectory: root.appendingPathComponent("addons"),
            textEncoderDirectory: root.appendingPathComponent("te"),
            tokenizerDirectory: root.appendingPathComponent("tok"))
    }

    // MARK: - MAT

    func testMATGate() throws {
        let report = MaterializationConformance.check(
            freshConfiguration: FastFramesConfiguration(), satisfiedConfiguration: try Self.satisfied())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testFourSourcesAndNoAppleMirror() {
        let src = FastFramesConfiguration().weightSources
        XCTAssertEqual(src.map(\.role), ["dit", "addons", "text-encoder", "tokenizer"])
        // AB-D-0259: Apple's AMLR files come from Apple's own repo, never a mirror
        XCTAssertEqual(src.first { $0.role == "addons" }?.repo, "apple/FastFrames-S")
        XCTAssertFalse(src.contains { $0.repo.hasPrefix("mlx-community/") && $0.role == "addons" })
        XCTAssertEqual(FastFramesConfiguration().missingWeightSources(storeRoot: nil).count, 4, "fresh machine")
    }

    func testPartialExplicitPathsReportOnlyTheirSource() throws {
        var cfg = try Self.satisfied()
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        cfg.ditFile = cfg.ditFile!.deletingLastPathComponent().appendingPathComponent("absent.safetensors")
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["dit"])
    }

    func testAddonsResolveFlatOrNested() throws {
        let cfg = try Self.satisfied()
        XCTAssertTrue(cfg.resolvedPaths()!.refiner.path.hasSuffix("addons/fastframes-s/kv_refiner.safetensors"))
        let flat = try Self.tempDir()
        try Data([0]).write(to: flat.appendingPathComponent("kv_refiner.safetensors"))
        try Data([0]).write(to: flat.appendingPathComponent("pixel_decoder.safetensors"))
        var c2 = cfg
        c2.addonsDirectory = flat
        XCTAssertEqual(c2.resolvedPaths()!.decoder, flat.appendingPathComponent("pixel_decoder.safetensors"))
        XCTAssertTrue(c2.missingWeightSources(storeRoot: nil).isEmpty)
    }

    func testConfigurationCodableCarriesNoHostPaths() throws {
        let data = try JSONEncoder().encode(try Self.satisfied())
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "{}")
        let back = try JSONDecoder().decode(FastFramesConfiguration.self, from: data)
        XCTAssertNil(back.ditFile)
        XCTAssertNil(back.modelsRootDirectory)
    }

    // MARK: - licence

    func testLicenceDeclaresAMLRWithNotices() {
        let lic = FastFramesPackage.manifest.license
        XCTAssertEqual(lic.weightLicenses, [.apache2, FastFramesLicense.amlr])
        XCTAssertEqual(lic.portCodeLicense, FastFramesLicense.appleSampleCode)
        XCTAssertNil(lic.portCodeLicenseLackingNotice, "the port-code licence must carry its notice")
        XCTAssertTrue(lic.missingRequiredDisplayNotices.isEmpty)
        let amlrNotice = lic.notices?.first { $0.license == FastFramesLicense.amlr }
        XCTAssertNotNil(amlrNotice)
        // the AMLR attribution sentence, verbatim
        XCTAssertTrue(amlrNotice!.text.contains(
            "Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model "
                + "License Agreement."))
        XCTAssertFalse(LicensePolicy.permissiveOnly.admits(FastFramesLicense.amlr), "never a shippable licence")
    }

    func testBlockingPermissiveHostRefuses() async {
        let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)
        do {
            _ = try await engine.register(FastFramesPackage.registration, configuration: FastFramesConfiguration())
            XCTFail("a blocking permissive-only host must refuse the AMLR weights")
        } catch EngineError.licenseRejected(let result) {
            XCTAssertNotEqual(result, .admitted)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testAdvisoryHostRegistersAndRecordsTheFindings() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(FastFramesPackage.registration, configuration: FastFramesConfiguration())
        let licences = await engine.licenseAdvisories.map(\.license)
        XCTAssertTrue(licences.contains(FastFramesLicense.amlr), "\(licences)")
    }

    // MARK: - FIT

    func testFITGate() {
        let report = FootprintConformance.check(
            manifest: FastFramesPackage.manifest, configuration: FastFramesConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN

    func testCANPreCancelledRun() async {
        let report = await CancellationConformance.checkRun(
            package: FastFramesPackage(configuration: FastFramesConfiguration()),
            request: T2VRequest(prompt: "x"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: FastFramesPackage.manifest,
            // after every latent frame in the rollout's onFrame seam, with RunProgress(.denoise) there
            posture: .cadence([.init(phase: .denoise, unit: .frame, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - request planning

    func testFramesMapToLatentFrames() throws {
        func latent(_ n: Int?) throws -> Int { try FastFramesPackage.plan(T2VRequest(prompt: "p", numFrames: n)).latentFrames }
        XCTAssertEqual(try latent(nil), 21)
        XCTAssertEqual(try latent(81), 21)
        XCTAssertEqual(try latent(78), 21)
        XCTAssertEqual(try latent(77), 20)
        XCTAssertEqual(try latent(5), 2)
        XCTAssertEqual(try latent(1), 1)
        // every count decodes at least as many frames as asked (4L − 3 ≥ n), and no more than 3 extra
        for n in 1...81 {
            let l = try latent(n)
            XCTAssertGreaterThanOrEqual(4 * l - 3, n)
            XCTAssertLessThan(4 * l - 3 - n, 4)
        }
    }

    func testPlanDefaultsAndSeed() throws {
        let p = try FastFramesPackage.plan(T2VRequest(prompt: "p", seed: 9))
        XCTAssertEqual(p, .init(frames: 81, latentFrames: 21, fps: 16, seed: 9))
        XCTAssertEqual(try FastFramesPackage.plan(T2VRequest(prompt: "p", fps: 24)).fps, 24)
        // native values are accepted, not refused
        XCTAssertNoThrow(try FastFramesPackage.plan(T2VRequest(
            prompt: "p", width: 832, height: 480, steps: 2, guidanceScale: 1)))
    }

    func testUnsupportedFieldsAreRefusedLegibly() {
        let bad: [T2VRequest] = [
            T2VRequest(prompt: "p", negativePrompt: "blurry"),
            T2VRequest(prompt: "p", initImage: Image(format: .png, data: Data(), width: 1, height: 1)),
            T2VRequest(prompt: "p", numFrames: 82),
            T2VRequest(prompt: "p", numFrames: 0),
            T2VRequest(prompt: "p", width: 1280),
            T2VRequest(prompt: "p", height: 720),
            T2VRequest(prompt: "p", steps: 8),
            T2VRequest(prompt: "p", guidanceScale: 5),
            T2VRequest(prompt: "p", includeAudio: true),
        ]
        for r in bad {
            XCTAssertThrowsError(try FastFramesPackage.plan(r)) { e in
                guard case PackageError.unsupportedRequestFeature(let why) = e else { return XCTFail("\(e)") }
                XCTAssertTrue(why.hasPrefix("FastFrames-S: "))
            }
        }
    }

    // MARK: - load refusal

    @InferenceActor
    func testLoadRefusesInsteadOfDownloading() async throws {
        var cfg = try Self.satisfied()
        cfg.ditFile = nil  // no explicit DiT and no store → not provisioned
        let pkg = FastFramesPackage(configuration: cfg)
        do {
            try await pkg.load()
            XCTFail("load must refuse when a component is absent")
        } catch let e as FastFramesPackageError {
            guard case .weightsNotProvisioned = e else { return XCTFail("\(e)") }
            XCTAssertTrue(e.description.contains(FastFramesConfiguration.ditRepo))
        }
    }
}
