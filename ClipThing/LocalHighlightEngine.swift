import AVFoundation
import CoreImage
import Foundation
import Metal
import MLX
import MLXLMCommon
import MLXVLM
import os

/// One process-wide inference owner, including across multiple app windows.
actor LocalHighlightEngine {
    static let shared = LocalHighlightEngine()
    private var running = false

    func highlights(
        in url: URL, duration: Double,
        progress: @Sendable @escaping (String) async -> Void
    ) async throws -> [HighlightSpan] {
        guard !running else { throw EngineError.alreadyRunning }
        running = true
        defer { running = false }
        let windows = try HighlightPlan.windows(duration: duration)
        #if targetEnvironment(simulator)
        throw EngineError.simulator
        #else
        guard MTLCreateSystemDefaultDevice() != nil else { throw EngineError.unsupportedDevice }
        // A conservative app guard, not a claim that all 6 GB devices have been validated.
        guard ProcessInfo.processInfo.physicalMemory >= 6_000_000_000 else {
            throw EngineError.insufficientMemory
        }
        guard let directory = Bundle.main.resourceURL?.appendingPathComponent("SmolVLM2"),
              FileManager.default.fileExists(atPath: directory.appendingPathComponent("model.safetensors").path) else {
            throw EngineError.missingModel
        }
        try Task.checkCancellation()
        await progress("Loading on-device model…")
        GPU.set(cacheLimit: 32 * 1024 * 1024)
        defer {
            Stream().synchronize()
            GPU.clearCache()
        }
        try InferenceMemory.checkHeadroom(required: 2_000 * 1024 * 1024)
        // Local directory loading also keeps the tokenizer offline; never pass a Hub model ID here.
        let configuration = ModelConfiguration(directory: directory, extraEOSTokens: ["<end_of_utterance>"])
        // A private registry avoids changing shared factory behavior or racing another loader.
        let factory = VLMModelFactory(typeRegistry: ModelTypeRegistry(creators: [
            "idefics3": { url in
                let config = try JSONDecoder().decode(Idefics3Configuration.self,
                    from: Data(contentsOf: url))
                return MemoryEfficientIdefics3(config)
            }
        ]), processorRegistry: VLMProcessorTypeRegistry.shared, modelRegistry: VLMRegistry.shared)
        let container = try await factory.loadContainer(configuration: configuration)
        let geometry = try VideoModelGeometry(directory: directory)
        var candidates: [HighlightSpan] = []
        for (index, window) in windows.enumerated() {
            try Task.checkCancellation()
            await progress("Watching section \(index + 1) of \(windows.count)…")
            let output = try await container.perform { context in
                let input = try await TimestampedVideoInput.prepare(
                    url: url, window: window, geometry: geometry, context: context)
                try Task.checkCancellation()
                let parameters = GenerateParameters(maxTokens: 180, temperature: 0, prefillStepSize: 128)
                var lowMemory = false
                // The synchronous visitor stays in this task and synchronizes the GPU before returning.
                // Unlike an unstructured streaming producer, cancellation cannot leave generation running.
                let result = try MLXLMCommon.generate(input: input, parameters: parameters, context: context) {
                    (tokens: [Int]) in
                    lowMemory = InferenceMemory.isLow
                    return Task.isCancelled || lowMemory || tokens.count >= 180 ? .stop : .more
                }
                try Task.checkCancellation()
                if lowMemory { throw InferenceMemory.PressureError() }
                return result.output
            }
            candidates += try HighlightPlan.parse(output, window: window)
            GPU.clearCache()
        }
        try Task.checkCancellation()
        // Container is local to this job; weights are released before export/preview.
        return HighlightPlan.merge(candidates, duration: duration)
        #endif
    }
}

struct VideoModelGeometry: Sendable {
    let imageSize: Int
    let imageTokens: Int
    let imageTokenID: Int

    init(directory: URL) throws {
        struct Configuration: Decodable {
            struct Vision: Decodable { let image_size: Int; let patch_size: Int }
            let vision_config: Vision
            let scale_factor: Int
            let image_token_id: Int
        }
        let config = try JSONDecoder().decode(Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        imageSize = config.vision_config.image_size
        guard imageSize == 384, config.vision_config.patch_size == 14, config.scale_factor == 3 else {
            throw EngineError.modelMismatch
        }
        let grid = imageSize / config.vision_config.patch_size / config.scale_factor
        imageTokens = grid * grid // 81 for this checkpoint, not the library processor's default 64.
        imageTokenID = config.image_token_id
    }
}

private enum TimestampedVideoInput {
    static func prepare(url: URL, window: HighlightSpan, geometry: VideoModelGeometry,
                        context: ModelContext) async throws -> LMInput {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 768, height: 768)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        defer { generator.cancelAllCGImageGeneration() }
        let renderContext = CIContext(options: [.cacheIntermediates: false])
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let edge = geometry.imageSize
        var frames: [MLXArray] = []
        var framePrompt = "Video frames in chronological order. Timestamps are seconds from the start of this section.\n"
        for seconds in HighlightPlan.sampleTimes(in: window) {
            try Task.checkCancellation()
            let result = try await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 600))
            try Task.checkCancellation()
            let relative = min(window.duration, max(0, result.actualTime.seconds - window.start))
            // Render RGB once with explicit sRGB conversion. Normalize [0,255] -> [-1,1].
            // Match the checkpoint's video path: one 384x384 image per frame, no image tiling.
            let original = CIImage(cgImage: result.image)
            let resized = MediaProcessing.resampleLanczos(original, to: CGSize(width: edge, height: edge))
            var bytes = [UInt8](repeating: 0, count: edge * edge * 4)
            bytes.withUnsafeMutableBytes { buffer in
                renderContext.render(resized, toBitmap: buffer.baseAddress!, rowBytes: edge * 4,
                    bounds: CGRect(x: 0, y: 0, width: edge, height: edge), format: .RGBA8, colorSpace: colorSpace)
            }
            let rgba = MLXArray(bytes, [1, edge, edge, 4])
            let rgb = (rgba[0..., 0..., 0..., ..<3].asType(.float32) / 127.5 - 1).asType(.float16)
            eval(rgb)
            frames.append(rgb)
            let timestamp = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), relative)
            framePrompt += "Frame at \(timestamp) seconds: <fake_token_around_image><global-img>"
            framePrompt += String(repeating: "<image>", count: geometry.imageTokens)
            framePrompt += "<fake_token_around_image>\n"
        }
        let sectionLength = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), window.duration)
        let instructions = """
        Select meaningful action worth keeping from this \(sectionLength)-second section.
        Preserve complete events, including setup, action, and outcome. In sports keep the throw,
        ball flight, catch, and finish together. Do not select only celebrations or spectators.
        If an interesting action is already happening at the start, start at 0. If still happening
        at the end, end at \(sectionLength). Select up to 3 intervals; use an empty array if none.
        Return ONLY JSON: {"highlights":[{"start":0.0,"end":5.0}]}
        Replace the example with observed times in seconds relative to THIS section, between 0
        and \(sectionLength). Do not use timestamps from any other video or add explanations.
        """
        // Use the checkpoint's own chat template; image placeholders must stay literal special tokens.
        let messages: [MLXLMCommon.Message] = [
            ["role": "user", "content": [["type": "text", "text": framePrompt + "\n" + instructions]]]
        ]
        let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
        guard tokens.filter({ $0 == geometry.imageTokenID }).count == frames.count * geometry.imageTokens,
              tokens.count + 180 < 8192 else { throw EngineError.modelMismatch }
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)),
            image: .init(pixels: concatenated(frames, axis: 0)))
    }
}

enum InferenceMemory {
    // Advisory reserve, not a guaranteed peak-memory prediction. Check afresh at each
    // bounded unit of work; physical RAM does not describe the app's jetsam allowance.
    static var isLow: Bool { os_proc_available_memory() < 384 * 1024 * 1024 }

    static func checkHeadroom(required: Int = 384 * 1024 * 1024) throws {
        guard os_proc_available_memory() >= required else { throw PressureError() }
    }

    struct PressureError: LocalizedError {
        var errorDescription: String? {
            "There isn't enough app memory to finish on-device analysis safely. Try again after closing other apps. This model may be too large for this device."
        }
    }
}

private enum EngineError: LocalizedError {
    case alreadyRunning, simulator, unsupportedDevice, insufficientMemory, missingModel, modelMismatch
    var errorDescription: String? {
        switch self {
        case .alreadyRunning: "Another video is being analyzed. Wait for it to finish."
        case .simulator: "On-device analysis requires a real iPhone or iPad. MLX cannot run in the simulator."
        case .unsupportedDevice: "This device doesn't support the model's Metal GPU runtime."
        case .insufficientMemory: "This model needs a device with at least 6 GB of RAM."
        case .missingModel: "The bundled model is missing. Rebuild the app with the Prepare Model build step enabled."
        case .modelMismatch: "The bundled model and video processor don't match. Rebuild the app to restore the pinned model."
        }
    }
}
