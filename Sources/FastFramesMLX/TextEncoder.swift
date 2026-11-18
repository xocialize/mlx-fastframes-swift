// Prompt → zero-padded umT5-XXL features `[1, 512, 4096]` through wan-core's encoder (the Wan2.1 umT5 weights, as
// converted for the mlx-video Wan family: `t5_encoder.safetensors` + the checkpoint's `config.json`). Upstream
// (`fastframes/wan/text_encoder.py`) zeroes every position past the prompt's tokens; this does the same.

import Foundation
import MLX
import MLXNN
import Tokenizers
import WanCore

public final class FastFramesTextEncoder {
    let encoder: UMT5EncoderModel
    let tokenizer: any Tokenizer

    /// - wanDirectory: a converted Wan checkpoint dir holding `t5_encoder.safetensors` and `config.json`
    /// - tokenizerDirectory: a local `google/umt5-xxl` snapshot (`tokenizer.json`, `tokenizer_config.json`)
    public init(wanDirectory: URL, tokenizerDirectory: URL) async throws {
        let config = try WanConfig.load(from: wanDirectory.appendingPathComponent("config.json"))
        let model = UMT5EncoderModel.fromConfig(config)
        let weights = try WeightLoader.loadVerifiedSafetensors(
            url: wanDirectory.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        self.encoder = model
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    }

    public func encode(_ prompt: String) -> MLXArray {
        let valid = encodeText(encoder: encoder, tokenizer: tokenizer, prompt: prompt, textLen: FF.textLen)
        let pad = FF.textLen - valid.dim(0)
        let full = pad > 0
            ? concatenated([valid, MLXArray.zeros([pad, valid.dim(1)], dtype: valid.dtype)], axis: 0) : valid
        return full[.newAxis].asType(.float32)
    }
}
