import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

// The CI executable tests math on tiny tensors, not iOS memory policy.
enum InferenceMemory {
    static func checkHeadroom() throws { }
    static func prefillChunkSize(maximum: Int) -> Int { maximum }
}

Device.setDefault(device: Device(.cpu))
let configuration = try JSONDecoder().decode(Idefics3Configuration.self, from: Data("""
{
  "model_type":"smolvlm", "scale_factor":2, "image_token_id":31, "vocab_size":32,
  "text_config": {
    "model_type":"llama", "hidden_size":32, "num_hidden_layers":2,
    "intermediate_size":64, "num_attention_heads":4, "num_key_value_heads":4,
    "rms_norm_eps":0.00001, "vocab_size":32, "rope_theta":10000,
    "tie_word_embeddings":true
  },
  "vision_config": {
    "model_type":"smolvlm_vision", "hidden_size":32, "num_hidden_layers":2,
    "intermediate_size":64, "num_attention_heads":4, "image_size":28, "patch_size":7
  }
}
""".utf8))
let reference = Idefics3(configuration)
let bounded = MemoryEfficientIdefics3(configuration)
// Identical weights also verify that the adaptation preserves module parameter paths.
try bounded.update(parameters: reference.parameters(), verify: .all)
eval(reference, bounded)
let tokens = MLXArray([1, 31, 31, 31, 31, 2, 31, 31, 31, 31, 3, 4]).expandedDimensions(axis: 0)
let pixels = (MLXArray(0..<(2 * 28 * 28 * 3)).reshaped(2, 28, 28, 3).asType(.float32) / 5000).asType(.float16)
let input = LMInput(text: .init(tokens: tokens), image: .init(pixels: pixels))

func logits(_ result: PrepareResult) -> MLXArray {
    guard case .logits(let output) = result else { fatalError("Expected prefill logits") }
    return output.logits[0..., -1, 0...]
}
func close(_ actual: MLXArray, _ expected: MLXArray) {
    let error = abs(actual - expected).max().item(Float.self)
    precondition(error.isFinite && error < 0.01, "Chunked inference changed logits: \(error)")
}
for chunk in [1, 5, 128] {
    let referenceCache = reference.newCache(parameters: nil)
    let boundedCache = bounded.newCache(parameters: nil)
    let expected = logits(try reference.prepare(input, cache: referenceCache, windowSize: nil))
    let actual = logits(try bounded.prepare(input, cache: boundedCache, windowSize: chunk))
    close(actual, expected)
    precondition(boundedCache.allSatisfy { $0.offset == tokens.dim(1) })
    // Decode using the prefilled caches: catches lost history and incorrect RoPE offsets.
    for token in [5, 6, 7] {
        let next = MLXArray([token]).expandedDimensions(axis: 0)
        close(bounded(next, cache: boundedCache), reference(next, cache: referenceCache))
    }
}
print("Sequential vision, chunked prefill, and cached decoding match upstream.")
