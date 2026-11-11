import AVFoundation
import Foundation
import MediaIntelligence
import OSLog

/// One analysis owner across app windows. Cancellation retains ownership until Apple returns.
actor AppleHighlightEngine {
    static let shared = AppleHighlightEngine()
    private var running = false
    private let logger = Logger(subsystem: "ClipThing", category: "AppleHighlights")

    func highlights(
        in url: URL, duration: Double,
        progress: @Sendable @escaping (String) async -> Void
    ) async throws -> [HighlightSpan] {
        guard !running else { throw AnalysisError.alreadyRunning }
        guard duration.isFinite, duration > 0 else { throw AnalysisError.invalidDuration }
        running = true
        defer { running = false }
        try Task.checkCancellation()
        await progress("Finding highlights with Apple...")
        try Task.checkCancellation()
        let asset = MediaIntelligenceVideoAsset(
            id: MediaIntelligenceVideoAsset.ID(UUID().uuidString), kind: .url(url))
        let result = try await VideoAnalyzer.shared.analyze(asset, for: HighlightAnalysisRequest())
        try Task.checkCancellation()
        let analysis = try result.get()
        logger.debug("Apple returned \(analysis.highlights.count) highlights and \(analysis.levels.count) engagement segments")
        let spans = analysis.highlights.compactMap { range -> HighlightSpan? in
            guard range.isValid, !range.isIndefinite else { return nil }
            let start = range.start.seconds
            let end = range.end.seconds
            logger.debug("Apple highlight: \(start) to \(end) seconds")
            return HighlightSpan(start: start, end: end)
        }
        // Preserve Apple's actual decisions. No padding chains, LLM scores or duration caps.
        return HighlightPlan.normalized(spans, duration: duration)
    }
}

private enum AnalysisError: LocalizedError {
    case alreadyRunning, invalidDuration

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: "Another video is being analyzed. Wait for it to finish."
        case .invalidDuration: "This video doesn't have a usable duration. Try another video."
        }
    }
}
