import Foundation

struct HighlightSpan: Equatable, Sendable {
    let start: Double
    let end: Double
    var duration: Double { end - start }
}

enum HighlightPlan {
    /// Validate and clamp Apple ranges without changing separate event boundaries.
    static func normalized(_ spans: [HighlightSpan], duration: Double) -> [HighlightSpan] {
        guard duration.isFinite, duration > 0 else { return [] }
        let valid = spans.filter { $0.start.isFinite && $0.end.isFinite && $0.end > $0.start }
            .map { HighlightSpan(start: max(0, $0.start), end: min(duration, $0.end)) }
            .filter { $0.duration > 0 }
            .sorted {
                if $0.start != $1.start { return $0.start < $1.start }
                return $0.end < $1.end
            }
        var unique: [HighlightSpan] = []
        for span in valid where unique.last != span { unique.append(span) }
        return unique
    }
}
