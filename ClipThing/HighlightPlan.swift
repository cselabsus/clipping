import Foundation

/// Pure timeline logic, independent of the ML runtime and UI.
struct HighlightSpan: Equatable, Sendable {
    let start: Double
    let end: Double
    var duration: Double { end - start }
}

struct HighlightCandidate: Equatable, Sendable {
    let span: HighlightSpan
    let score: Int
}

enum HighlightPlan {
    static let windowLength = 12.0
    static let overlap = 4.0
    static let framesPerSecond = 2.0
    static let maximumDuration = 600.0
    static let maximumClipDuration = 20.0
    static let maximumClipCount = 3
    static let contextPadding = 3.0

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
        let score: Int
    }

    static func parse(_ output: String, window: HighlightSpan) throws -> [HighlightCandidate] {
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
                  candidate.end <= window.duration + 0.05, (1...5).contains(candidate.score) else {
                throw PlanError.invalidResponse
            }
            return HighlightCandidate(
                span: HighlightSpan(start: window.start + candidate.start,
                                    end: min(window.end, window.start + candidate.end)),
                score: candidate.score)
        }
    }

    /// Rank before adding context. Padding must never create a relationship between events.
    static func select(_ candidates: [HighlightCandidate], duration: Double) -> [HighlightSpan] {
        guard duration.isFinite, duration > 0 else { return [] }
        // Short videos may contain just one event. Longer videos get a bounded digest.
        let budget = min(duration, min(45, max(12, duration * 0.35)))
        let ranked = candidates.filter {
            $0.score >= 4 && $0.score <= 5 && $0.span.start.isFinite && $0.span.end.isFinite
                && $0.span.start >= 0 && $0.span.end <= duration && $0.span.duration > 0
                && $0.span.duration <= maximumClipDuration
        }.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.span.start != $1.span.start { return $0.span.start < $1.span.start }
            return $0.span.end > $1.span.end
        }
        var selected: [HighlightSpan] = []
        for anchor in ranked {
            if selected.count == maximumClipCount { break }
            var core = anchor.span
            // Join only substantial raw overlap with the ORIGINAL anchor. Never follow
            // an expanding overlap chain, and never let a joined clip exceed the cap.
            for other in ranked {
                let shared = intersection(anchor.span, other.span)
                guard shared > 0, shared >= min(anchor.span.duration, other.span.duration) * 0.5 else {
                    continue
                }
                let union = HighlightSpan(start: min(core.start, other.span.start),
                                          end: max(core.end, other.span.end))
                if padded(union, duration: duration).duration <= maximumClipDuration {
                    core = union
                }
            }
            // Keep the higher-ranked version of overlapping detections rather than
            // exporting duplicates or slicing an event to fill the remaining budget.
            guard !selected.contains(where: { intersection($0, core) > 0 }),
                  padded(core, duration: duration).duration <= maximumClipDuration else { continue }
            let proposed = selected + [core]
            let clips = addContext(proposed, duration: duration)
            guard clips.reduce(0, { $0 + $1.duration }) <= budget + 0.000001 else { continue }
            selected = proposed
        }
        return addContext(selected, duration: duration)
    }

    private static func intersection(_ a: HighlightSpan, _ b: HighlightSpan) -> Double {
        max(0, min(a.end, b.end) - max(a.start, b.start))
    }

    private static func padded(_ span: HighlightSpan, duration: Double) -> HighlightSpan {
        HighlightSpan(start: max(0, span.start - contextPadding),
                      end: min(duration, span.end + contextPadding))
    }

    private static func addContext(_ cores: [HighlightSpan], duration: Double) -> [HighlightSpan] {
        let ordered = cores.sorted { $0.start < $1.start }
        var clips = ordered.map { padded($0, duration: duration) }
        for index in ordered.indices.dropFirst() {
            if clips[index - 1].end > clips[index].start {
                // Divide only the gap's padding. Preserve both raw events, as two clips.
                let boundary = (ordered[index - 1].end + ordered[index].start) / 2
                clips[index - 1] = .init(start: clips[index - 1].start, end: boundary)
                clips[index] = .init(start: boundary, end: clips[index].end)
            }
        }
        return clips
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
