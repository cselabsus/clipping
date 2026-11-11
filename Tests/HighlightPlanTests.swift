import Foundation

@main
struct HighlightPlanTests {
    static func main() {
        let events: [HighlightSpan] = [.init(start: 3, end: 7), .init(start: 12, end: 16)]
        precondition(HighlightPlan.normalized(events, duration: 118) == events)
        // Touching and overlapping ranges stay separate; no transitive union of clips.
        let overlaps: [HighlightSpan] = [.init(start: 0, end: 12), .init(start: 8, end: 20),
                                        .init(start: 20, end: 30)]
        precondition(HighlightPlan.normalized(overlaps, duration: 118) == overlaps)
        let invalid: [HighlightSpan] = [.init(start: .nan, end: 5), .init(start: 5, end: .infinity),
                                       .init(start: 9, end: 4), .init(start: 4, end: 4),
                                       .init(start: 119, end: 125), .init(start: -5, end: -1)]
        precondition(HighlightPlan.normalized(invalid, duration: 118).isEmpty)
        let bounds: [HighlightSpan] = [.init(start: 110, end: 130), .init(start: -3, end: 6),
                                      .init(start: -3, end: 6)]
        precondition(HighlightPlan.normalized(bounds, duration: 118) ==
                     [.init(start: 0, end: 6), .init(start: 110, end: 118)])
        // Complete plays are no longer cut by the old 20-second cap or total-length budget.
        let longPlay: [HighlightSpan] = [.init(start: 10, end: 39)]
        precondition(HighlightPlan.normalized(longPlay, duration: 118) == longPlay)
        precondition(HighlightPlan.normalized([], duration: 118).isEmpty)
        for duration in [0, -1, Double.nan, Double.infinity] {
            precondition(HighlightPlan.normalized(events, duration: duration).isEmpty)
        }
        print("Apple highlight boundary tests passed.")
    }
}
