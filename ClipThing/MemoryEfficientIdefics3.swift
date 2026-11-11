// Adapted from ml-explore/mlx-swift-lm 2.29.3, Libraries/MLXVLM/Models/Idefics3.swift.
// Copyright (c) 2024 ml-explore. MIT license; full notice in ThirdPartyNotices.txt.
// Original file created by SHUHONG WU on 12/13/24.
// Local changes: sequential vision encoding, bounded prefill, last-position logits,
// cancellation and process-memory checks. Parameter names/shapes remain unchanged.

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

private class Idefics3MLP: Module, UnaryLayer {
    @ModuleInfo var proj: Linear
    init(_ config: Idefics3Configuration) {
        let inputSize = config.visionConfig.hiddenSize * (config.scaleFactor * config.scaleFactor)
        let outputSize = config.textConfig.hiddenSize
        self._proj.wrappedValue = Linear(inputSize, outputSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = proj(x)
        return out
    }
}

private class Idefics3Connector: Module {
    let scaleFactor: Int
    @ModuleInfo(key: "modality_projection") var modalityProjection: Idefics3MLP

    init(_ config: Idefics3Configuration) {
        self.scaleFactor = config.scaleFactor
        self._modalityProjection.wrappedValue = Idefics3MLP(config)
    }

    func pixelShuffle(_ x: MLXArray, scaleFactor: Int) -> MLXArray {
        let B = x.dim(0)
        let seq = x.dim(1)
        let embed_dim = x.dim(2)
        let side = Int(Double(seq).squareRoot())

        var reshaped = x.reshaped(B, side, side, embed_dim)
        reshaped =
            reshaped
            .reshaped(B, side, side / scaleFactor, embed_dim * scaleFactor)
        reshaped = reshaped.transposed(0, 2, 1, 3)
        reshaped =
            reshaped
            .reshaped(
                B,
                side / scaleFactor,
                side / scaleFactor,
                embed_dim * (scaleFactor * scaleFactor)
            )
        reshaped = reshaped.transposed(0, 2, 1, 3)
        reshaped =
            reshaped
            .reshaped(
                B,
                seq / (scaleFactor * scaleFactor),
                embed_dim * (scaleFactor * scaleFactor)
            )
        return reshaped
    }

    func callAsFunction(_ imageHiddenStates: MLXArray) -> MLXArray {
        let shuffled = pixelShuffle(imageHiddenStates, scaleFactor: scaleFactor)
        let out = modalityProjection(shuffled)
        return out
    }
}

// MARK: - Language

private enum Language {
    fileprivate class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var q_proj: Linear
        @ModuleInfo(key: "k_proj") var k_proj: Linear
        @ModuleInfo(key: "v_proj") var v_proj: Linear
        @ModuleInfo(key: "o_proj") var o_proj: Linear
        @ModuleInfo(key: "rope") var ropeEmbed: RoPE

        init(_ config: Idefics3Configuration.TextConfiguration) {
            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads
            let headDim = dim / nHeads
            self.scale = pow(Float(headDim), -0.5)

            self._q_proj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._k_proj.wrappedValue = Linear(
                dim,
                nKVHeads * headDim,
                bias: false
            )
            self._v_proj.wrappedValue = Linear(
                dim,
                nKVHeads * headDim,
                bias: false
            )
            self._o_proj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            self._ropeEmbed.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache? = nil
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)
            var q = q_proj(x)
            var k = k_proj(x)
            var v = v_proj(x)

            q = q.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            q = ropeEmbed(q, offset: offset)
            k = ropeEmbed(k, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: q,
                keys: k,
                values: v,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            let final = o_proj(output)
            return final
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate_proj: Linear
        @ModuleInfo(key: "down_proj") var down_proj: Linear
        @ModuleInfo(key: "up_proj") var up_proj: Linear
        init(dim: Int, hiddenDim: Int) {
            self._gate_proj.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down_proj.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up_proj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let g = gate_proj(x)
            let r = down_proj(silu(g) * up_proj(x))
            return r
        }
    }

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(
            key: "post_attention_layernorm"
        ) var postAttentionLayerNorm: RMSNorm
        let mlp: MLP

        init(_ config: Idefics3Configuration.TextConfiguration) {
            self._selfAttn.wrappedValue = Attention(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            self.mlp = MLP(
                dim: config.hiddenSize,
                hiddenDim: config.intermediateSize
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let a = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + a
            let m = mlp(postAttentionLayerNorm(h))
            let out = h + m
            return out
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        var layers: [TransformerBlock]
        let norm: RMSNorm
        let config: Idefics3Configuration.TextConfiguration
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        }
        var headDim: MLX.IntOrPair {
            .init(config.hiddenSize / config.numAttentionHeads)
        }

        init(_ config: Idefics3Configuration.TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )
            self.layers = (0 ..< config.numHiddenLayers)
                .map { _ in TransformerBlock(config) }
            self.norm = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            let lmHeadNeeded = !config.tieWordEmbeddings
            if lmHeadNeeded {
                self._lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabSize,
                    bias: false
                )
            }
        }

        func getEmbeddings(for inputIds: MLXArray) -> MLXArray {
            let e = embedTokens(inputIds)
            return e
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputs_embeds: MLXArray? = nil,
            lastLogitOnly: Bool = false
        ) -> LMOutput {
            let h: MLXArray
            if let inputs_embeds = inputs_embeds {
                h = inputs_embeds.asType(norm.weight.dtype)
            } else if let inputs = inputs {
                h = embedTokens(inputs)
            } else {
                fatalError(
                    "At least one of inputs or inputs_embeds must be provided."
                )
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)
            var x = h
            for (i, layer) in layers.enumerated() {
                let c = i < (cache?.count ?? 0) ? cache![i] : nil
                x = layer(x, mask: mask, cache: c)
            }

            // Only the final prompt position is used to select the next token.
            if lastLogitOnly { x = x[0..., (x.dim(1) - 1)...] }
            x = norm(x)
            let out = lmHead != nil ? lmHead!(x) : embedTokens.asLinear(x)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            // filter out rotary_emb.inv_freq
            return
                weights
                .filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        }
    }
}

// MARK: - Vision

private enum Vision {
    static func checkArrayShape(_ arr: MLXArray) -> Bool {
        if arr.ndim != 4 { return false }
        let (o, h, w, _) = (arr.dim(0), arr.dim(1), arr.dim(2), arr.dim(3))
        return (o >= h && o >= w && h == w)
    }

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var q_proj: Linear
        @ModuleInfo(key: "k_proj") var k_proj: Linear
        @ModuleInfo(key: "v_proj") var v_proj: Linear
        @ModuleInfo(key: "out_proj") var o_proj: Linear

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            let headDim = config.hiddenSize / config.numAttentionHeads
            self.scale = pow(Float(headDim), -0.5)
            self._q_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._k_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._v_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._o_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
            -> MLXArray
        {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            let q = q_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )
            let k = k_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )
            let v = v_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, D)
            let final = o_proj(output)
            return final
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.fc1 = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: true
            )
            self.fc2 = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let out = fc2(activation(fc1(x)))
            return out
        }
    }

    fileprivate class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var self_attn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self._self_attn.wrappedValue = Attention(config)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
            self.mlp = MLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
            -> MLXArray
        {
            let h = x + self_attn(layerNorm1(x), mask: mask)
            let out = h + mlp(layerNorm2(h))
            return out
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]
        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.layers = (0 ..< config.numHiddenLayers)
                .map { _ in EncoderLayer(config) }
        }

        func callAsFunction(
            _ x: MLXArray, outputHiddenStates: Bool = false,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        )
            -> (
                MLXArray,
                [MLXArray]?
            )
        {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            var h = x
            for l in layers {
                h = l(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }
            return (h, encoderStates)
        }
    }

    fileprivate class VisionEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPositions: Int

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize)
            )
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            self.numPositions = numPatches
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches,
                dimensions: config.hiddenSize
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = patchEmbedding(x)
            patchEmbeddings = patchEmbeddings.flattened(start: 1, end: 2)
            let positionIds = MLXArray(0 ..< numPositions)[.newAxis, 0...]
            let posEmbedding = positionEmbedding(positionIds)
            let embeddings = patchEmbeddings + posEmbedding
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "embeddings") var embeddings: VisionEmbeddings
        @ModuleInfo(key: "encoder") var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm
        let config: Idefics3Configuration.VisionConfiguration

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.config = config
            self._embeddings.wrappedValue = VisionEmbeddings(config)
            self._encoder.wrappedValue = Encoder(config)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = true) -> (
            MLXArray,
            MLXArray,
            [MLXArray]?
        ) {
            let e = embeddings(x)
            let (encoded, hiddenStates) = encoder(
                e,
                outputHiddenStates: outputHiddenStates
            )
            let pooler_output = postLayernorm(encoded)
            return (pooler_output, e, hiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_embedding.weight") {
                    if Vision.checkArrayShape(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Model

final class MemoryEfficientIdefics3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: Vision.VisionModel
    @ModuleInfo(
        key: "language_model"
    ) private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "connector") private var connector: Idefics3Connector
    public let config: Idefics3Configuration

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { languageModel.headDim }

    public var loraLayers: [Module] {
        languageModel.layers
    }

    public init(_ config: Idefics3Configuration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue =
            Language
            .LanguageModel(config.textConfig)
        self._connector.wrappedValue = Idefics3Connector(config)
    }

    private func getInputEmbeddings(inputIds: MLXArray?, pixelValues: MLXArray?) throws -> MLXArray {
        if pixelValues == nil {
            guard let inputIds = inputIds else {
                fatalError("inputIds required if no pixelValues")
            }
            let inputs_embeds = languageModel.getEmbeddings(for: inputIds)
            return inputs_embeds
        }

        guard let inputIds = inputIds, let pixelValues = pixelValues else {
            fatalError("inputIds and pixelValues required")
        }

        let inputs_embeds = languageModel.getEmbeddings(for: inputIds)
        // Materialize each frame before starting the next. Otherwise MLX's lazy graph
        // can still execute all frames together and recreate the original memory peak.
        var features: [MLXArray] = []
        for frame in 0..<pixelValues.dim(0) {
            try Task.checkCancellation()
            try InferenceMemory.checkHeadroom()
            let feature = autoreleasepool {
                let (encoded, _, _) = visionModel(
                    pixelValues[frame..<frame + 1].asType(.float16),
                    outputHiddenStates: false)
                let projected = connector(encoded.asType(inputs_embeds.dtype))
                eval(projected)
                return projected
            }
            features.append(feature)
            GPU.clearCache()
        }
        let image_features = concatenated(features, axis: 0)

        let final = prepareInputsForMultimodal(
            imageFeatures: image_features,
            inputs_embeds: inputs_embeds,
            inputIds: inputIds
        )
        return final
    }

    // inputs_merger
    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray, inputs_embeds: MLXArray, inputIds: MLXArray
    ) -> MLXArray {
        // Assumes bs == 1
        // inputIds shape: (1, seq_len)
        // asArray(Int.self) -> [[Int]], take [0] to get [Int]
        let ids: [[Int]] = [inputIds.asArray(Int.self)]

        let inputIdArray: [Int] = ids[0]

        let imageTokenIndex = config.imageTokenIndex
        let imagePositions = inputIdArray.enumerated().compactMap {
            $1 == imageTokenIndex ? $0 : nil
        }

        var segments = [MLXArray]()
        var start_idx = 0

        let chunkSize = imageFeatures.shape[1]  // 64
        let chunkCount = imagePositions.count / chunkSize  // Should be imageFeatures.shape[0]
        let chunks = (0 ..< chunkCount).map { startIndex in
            let start = startIndex * chunkSize
            let end = start + chunkSize
            return Array(imagePositions[start ..< end])
        }

        for (chunkIndex, chunk) in chunks.enumerated() {
            let currentImage = imageFeatures[chunkIndex]

            for (i, pos) in chunk.enumerated() {
                if pos > start_idx {
                    segments.append(inputs_embeds[0, start_idx ..< pos])
                }
                segments.append(currentImage[i ..< i + 1])
                start_idx = pos + 1
            }
        }

        if start_idx < inputs_embeds.dim(1) {
            segments.append(inputs_embeds[0, start_idx...])
        }

        let finalEmbeds = concatenated(segments, axis: 0)
        return finalEmbeds.expandedDimensions(axis: 0)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels
        defer { Stream().synchronize() }
        let embeddings = try getInputEmbeddings(inputIds: inputIds, pixelValues: pixelValues)
        eval(embeddings)
        // Honor the prefill bound while retaining the full causal KV history.
        // The pinned upstream implementation ignores windowSize and prefills everything.
        let maximumChunkSize = max(1, min(windowSize ?? 128, 128))
        var result: LMOutput?
        var start = 0
        while start < embeddings.dim(1) {
            try Task.checkCancellation()
            try InferenceMemory.checkHeadroom()
            let chunkSize = InferenceMemory.prefillChunkSize(maximum: maximumChunkSize)
            let end = min(start + chunkSize, embeddings.dim(1))
            let chunk = languageModel(nil, cache: cache,
                inputs_embeds: embeddings[0..., start..<end], lastLogitOnly: true)
            eval(chunk.logits)
            // Force every layer's updated cache before discarding this chunk's graph.
            eval(cache.flatMap { $0.state })
            GPU.clearCache()
            result = chunk
            start = end
        }
        guard let result else { throw VLMError.processing("The video prompt is empty.") }
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let out = languageModel(inputs, cache: cache).logits
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Rename keys to match Python logic
        var renamed = [String: MLXArray]()
        for (k, v) in weights {
            var newKey = k
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            } else if newKey.hasPrefix("lm_head.") {
                newKey = "language_model." + newKey
            }
            renamed[newKey] = v
        }

        var final = [String: MLXArray]()
        for (k, v) in renamed {
            if k.hasPrefix("text_model.") {
                let suffix = String(k.dropFirst("text_model.".count))
                final["language_model." + suffix] = v
            } else {
                final[k] = v
            }
        }

        // Remove rotary_emb.inv_freq
        final = final.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        return final
    }
}

