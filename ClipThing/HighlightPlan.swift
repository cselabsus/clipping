import Foundation

/// Pure timeline logic, independent of the ML runtime and UI.
struct HighlightSpan: Equatable, Sendable {
    let start: Double
    let end: Double
    var duration: Double { end - start }
}

enum HighlightPlan {
    static let windowLength = 12.0
    static let overlap = 4.0
    static let framesPerSecond = 2.0
    static let maximumDuration = 600.0

    static func windows(duration: Double) throws -> [HighlightSpan] {
        guard duration.isFinite, duration > 0, duration <= maximumDuration else {
            throw PlanError.unsupportedDuration
        }
        var result: [HighlightSpan] = []
        var start = 0.0
        while start < duration {
            let end = min(start + windowLength, duration)
            result.append(.init(start: start, end: end))
            if end == duration { break }
            start += windowLength - overlap
        }
        return result
    }

    static func sampleTimes(in window: HighlightSpan) -> [Double] {
        let count = max(1, Int(ceil(window.duration * framesPerSecond)))
        // Sample bin centers, never the asset's exclusive end timestamp.
        return (0..<count).map { window.start + (Double($0) + 0.5) * window.duration / Double(count) }
    }

    private struct Response: Decodable {
        let highlights: [Candidate]
    }
    private struct Candidate: Decodable {
        let start: Double
        let end: Double
    }

    static func parse(_ output: String, window: HighlightSpan) throws -> [HighlightSpan] {
        var json = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if json.hasPrefix("```"), json.hasSuffix("```"), let newline = json.firstIndex(of: "\n") {
            json = String(json[json.index(after: newline)..<json.index(json.endIndex, offsetBy: -3)])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let data = json.data(using: .utf8),
              let response = try? JSONDecoder().decode(Response.self, from: data),
              response.highlights.count <= 3 else {
            throw PlanError.invalidResponse
        }
        // Reject hallucinated/reversed/out-of-window values instead of silently creating random clips.
        return try response.highlights.map { candidate in
            guard candidate.start.isFinite, candidate.end.isFinite,
                  candidate.start >= 0, candidate.end > candidate.start,
                  candidate.end <= window.duration + 0.05 else {
                throw PlanError.invalidResponse
            }
            return HighlightSpan(start: window.start + candidate.start,
                                 end: min(window.end, window.start + candidate.end))
        }
    }

    static func merge(_ candidates: [HighlightSpan], duration: Double) -> [HighlightSpan] {
        // Context padding and merging keep neighboring detections of a play together.
        let padded = candidates.map {
            HighlightSpan(start: max(0, $0.start - 3), end: min(duration, $0.end + 3))
        }.sorted { $0.start < $1.start }
        var result: [HighlightSpan] = []
        for span in padded {
            if let last = result.last, span.start <= last.end {
                result[result.count - 1] = .init(start: last.start, end: max(last.end, span.end))
            } else {
                result.append(span)
            }
        }
        return result
    }
}

enum PlanError: LocalizedError {
    case unsupportedDuration, invalidResponse
    var errorDescription: String? {
        switch self {
        case .unsupportedDuration: "Choose a video between a fraction of a second and 10 minutes long."
        case .invalidResponse: "The model couldn't produce reliable clip timestamps. No clips were exported. Try a shorter video."
        }
    }
}
