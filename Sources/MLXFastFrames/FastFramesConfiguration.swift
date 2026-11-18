import FastFramesMLX
import Foundation
import MLXToolKit

/// Init-time configuration for `FastFramesPackage` (C9). Per-request prompt, frame count and seed ride the canonical
/// `T2VRequest`.
///
/// **Weights come from four sources, none of them a mirror of Apple's files** (AB-D-0259: no AMLR weight mirror):
///
/// | role | repo | files | licence |
/// |---|---|---|---|
/// | `dit` | `mlx-community/Causal-Forcing-PP-framewise-2step-bf16` | `dit.safetensors` | Apache-2.0 |
/// | `addons` | `apple/FastFrames-S` (Apple's own repo) | `fastframes-s/{kv_refiner,pixel_decoder}.safetensors` | AMLR |
/// | `text-encoder` | `mlx-community/Bernini-R-1.3B-bf16` | `t5_encoder.safetensors`, `config.json` | Apache-2.0 |
/// | `tokenizer` | `google/umt5-xxl` | tokenizer files | Apache-2.0 |
///
/// The `dit` repo is the Causal Forcing++ `framewise-2step.pt` `generator_ema` converted to bf16 safetensors with
/// upstream key names (`zhuhz22/Causal-Forcing` ships only the pickle). **It is not published yet**; until it is, set
/// `ditFile` (or a store that already holds it). The umT5 encoder is the Wan2.1/2.2 umt5-xxl as the Wan family already
/// ships it (byte-identical to the file the V2 gate ran against).
///
/// Every explicit path wins over the store and never touches the network.
public struct FastFramesConfiguration: PackageConfiguration, ModelStorable {
    public static let ditRepo = "mlx-community/Causal-Forcing-PP-framewise-2step-bf16"
    public static let addonsRepo = "apple/FastFrames-S"
    public static let textEncoderRepo = "mlx-community/Bernini-R-1.3B-bf16"
    public static let tokenizerRepo = "google/umt5-xxl"

    static let ditFiles = ["dit.safetensors"]
    static let addonFiles = ["fastframes-s/kv_refiner.safetensors", "fastframes-s/pixel_decoder.safetensors"]
    static let textEncoderFiles = ["t5_encoder.safetensors", "config.json"]
    static let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

    /// Converted DiT (`dit.safetensors`); host-specific → excluded from `Codable`.
    public var ditFile: URL?
    /// Directory holding `kv_refiner.safetensors` and `pixel_decoder.safetensors` (flat, or under `fastframes-s/`).
    public var addonsDirectory: URL?
    /// Directory holding `t5_encoder.safetensors` + its `config.json`.
    public var textEncoderDirectory: URL?
    /// Local `google/umt5-xxl` snapshot.
    public var tokenizerDirectory: URL?
    /// Stamped by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(ditFile: URL? = nil, addonsDirectory: URL? = nil, textEncoderDirectory: URL? = nil,
                tokenizerDirectory: URL? = nil, modelsRootDirectory: URL? = nil) {
        self.ditFile = ditFile
        self.addonsDirectory = addonsDirectory
        self.textEncoderDirectory = textEncoderDirectory
        self.tokenizerDirectory = tokenizerDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: CodingKey {}
    public init(from decoder: Decoder) throws { self.init() }
    public func encode(to encoder: Encoder) throws { _ = encoder.container(keyedBy: CodingKeys.self) }

    // MARK: - resolution

    func storeDirectory(_ repo: String, storeRoot: URL? = nil) -> URL? {
        ModelStore(root: storeRoot ?? modelsRootDirectory).directory(for: repo)
    }

    /// Where each component is read from: the explicit path, else the store's flat repo directory.
    public struct ResolvedPaths: Sendable {
        public let dit: URL
        public let refiner: URL
        public let decoder: URL
        public let textEncoderDirectory: URL
        public let tokenizerDirectory: URL

        var all: [URL] {
            [dit, refiner, decoder, textEncoderDirectory.appendingPathComponent("t5_encoder.safetensors"),
             textEncoderDirectory.appendingPathComponent("config.json"),
             tokenizerDirectory.appendingPathComponent("tokenizer.json")]
        }
    }

    public func resolvedPaths(storeRoot: URL? = nil) -> ResolvedPaths? {
        guard let dit = ditFile ?? storeDirectory(Self.ditRepo, storeRoot: storeRoot)?
                .appendingPathComponent("dit.safetensors"),
              let addons = addonsDirectory ?? storeDirectory(Self.addonsRepo, storeRoot: storeRoot),
              let te = textEncoderDirectory ?? storeDirectory(Self.textEncoderRepo, storeRoot: storeRoot),
              let tok = tokenizerDirectory ?? storeDirectory(Self.tokenizerRepo, storeRoot: storeRoot)
        else { return nil }
        func addon(_ name: String) -> URL {
            let nested = addons.appendingPathComponent("fastframes-s").appendingPathComponent(name)
            return FileManager.default.fileExists(atPath: nested.path) ? nested : addons.appendingPathComponent(name)
        }
        return ResolvedPaths(dit: dit, refiner: addon("kv_refiner.safetensors"),
                             decoder: addon("pixel_decoder.safetensors"), textEncoderDirectory: te,
                             tokenizerDirectory: tok)
    }

    /// The first missing file, or nil when every component is on disk.
    func firstMissingFile(storeRoot: URL? = nil) -> String? {
        guard let paths = resolvedPaths(storeRoot: storeRoot) else { return "<no explicit paths and no model store>" }
        return paths.all.first { !FileManager.default.fileExists(atPath: $0.path) }?.path
    }
}

extension FastFramesConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "dit", repo: Self.ditRepo, revision: "main", matching: Self.ditFiles),
            WeightSource(role: "addons", repo: Self.addonsRepo, revision: "main", matching: Self.addonFiles),
            WeightSource(role: "text-encoder", repo: Self.textEncoderRepo, revision: "main",
                         matching: Self.textEncoderFiles),
            WeightSource(role: "tokenizer", repo: Self.tokenizerRepo, revision: "main", matching: Self.tokenizerFiles),
        ]
    }

    /// Explicit paths first (a component whose explicit path is complete is not missing; incomplete → its source is
    /// missing), then the default store probe for the rest.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func present(_ dir: URL?, _ files: [String], flatten: Bool = false) -> Bool? {
            guard let dir else { return nil }
            return files.allSatisfy { f in
                fm.fileExists(atPath: dir.appendingPathComponent(f).path)
                    || (flatten && fm.fileExists(atPath: dir.appendingPathComponent((f as NSString).lastPathComponent).path))
            }
        }
        let explicit: [String: Bool?] = [
            "dit": ditFile.map { fm.fileExists(atPath: $0.path) },
            "addons": present(addonsDirectory, Self.addonFiles, flatten: true),
            "text-encoder": present(textEncoderDirectory, Self.textEncoderFiles),
            "tokenizer": present(tokenizerDirectory, Self.tokenizerFiles),
        ]
        let storeMissing = Set(defaultMissingWeightSources(storeRoot: storeRoot).map(\.role))
        return weightSources.filter { src in
            if let e = explicit[src.role] ?? nil { return !e }
            return storeMissing.contains(src.role)
        }
    }
}

extension FastFramesConfiguration: WeightPrewarming {
    /// The resident set (DiT, refiner, decoder) plus the umT5 file the first request pages in.
    public var prewarmPaths: [URL] {
        guard let p = resolvedPaths() else { return [] }
        return [p.dit, p.refiner, p.decoder, p.textEncoderDirectory.appendingPathComponent("t5_encoder.safetensors")]
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }
}

extension FastFramesConfiguration: QuantConfigured {
    /// One lane: bf16 DiT and refiner, fp32 decoder, bf16 umT5 (upstream's own dtypes).
    public var quant: Quant { .bf16 }
}
