import Foundation

@main
struct HighlightPlanTests {
    static func main() throws {
        let windows = try HighlightPlan.windows(duration: 45)
        precondition(windows.first?.start == 0 && windows.last?.end == 45)
        precondition(windows.count == 6)
        for (previous, next) in zip(windows, windows.dropFirst()) {
            precondition(next.start < previous.end, "Windows must overlap, never skip footage")
        }
        for window in windows {
            let times = HighlightPlan.sampleTimes(in: window)
            precondition(times.count <= 24 && !times.isEmpty)
            precondition(times.allSatisfy { $0 >= window.start && $0 < window.end })
        }
        for duration in [0.01, 1, 12, 12.01, 600] {
            let spans = try HighlightPlan.windows(duration: duration)
            precondition(spans.last?.end == duration)
            precondition(!HighlightPlan.sampleTimes(in: spans.last!).isEmpty)
        }
        for invalid in [0, -1, Double.infinity, Double.nan, 601] {
            do { _ = try HighlightPlan.windows(duration: invalid); preconditionFailure("Accepted bad duration") }
            catch PlanError.unsupportedDuration { }
        }
        let section = HighlightSpan(start: 16, end: 28)
        let spans = try HighlightPlan.parse("{\"highlights\":[{\"start\":2,\"end\":9,\"score\":4}]}", window: section)
        precondition(spans == [.init(span: .init(start: 18, end: 25), score: 4)], "Section-relative times need an absolute offset")
        let fenced = try HighlightPlan.parse("```json\n{\"highlights\":[]}\n```", window: section)
        precondition(fenced.isEmpty)
        for bad in ["hello", "{\"highlights\":[{\"start\":9,\"end\":2,\"score\":4}]}",
                    "{\"highlights\":[{\"start\":-1,\"end\":2,\"score\":4}]}",
                    "{\"highlights\":[{\"start\":0,\"end\":45,\"score\":4}]}",
                    "{\"highlights\":[{\"start\":\"0:01\",\"end\":3,\"score\":4}]}"] {
            do { _ = try HighlightPlan.parse(bad, window: section); preconditionFailure("Accepted malformed model output") }
            catch PlanError.invalidResponse { }
        }
        func event(_ start: Double, _ end: Double, _ score: Int = 4) -> HighlightCandidate {
            .init(span: .init(start: start, end: end), score: score)
        }
        for bad in [
            #"{"highlights":[{"start":1,"end":4}]}"#,
            #"{"highlights":[{"start":1,"end":4,"score":0}]}"#,
            #"{"highlights":[{"start":1,"end":4,"score":6}]}"#,
            #"{"highlights":[{"start":1,"end":4,"score":"high"}]}"#,
            #"{"highlights":[{"start":1,"end":4,"score":4.5}]}"#
        ] {
            do { _ = try HighlightPlan.parse(bad, window: section); preconditionFailure("Accepted bad score") }
            catch PlanError.invalidResponse { }
        }
        // Overlapping detections of one pass can still retain setup, flight and catch.
        let play = HighlightPlan.select([event(9, 12), event(10, 18)], duration: 45)
        precondition(play == [.init(start: 6, end: 21)])
        let bounds = HighlightPlan.select([event(0, 2), event(44, 45)], duration: 45)
        precondition(bounds == [.init(start: 0, end: 5), .init(start: 41, end: 45)])

        // Padding overlap must NOT turn separate moments into a single clip.
        let separate = HighlightPlan.select([event(3, 7), event(12, 16)], duration: 118)
        precondition(separate == [.init(start: 0, end: 9.5), .init(start: 9.5, end: 19)])

        // Regression: even selecting every section cannot export the entire 118 seconds.
        let longWindows = try HighlightPlan.windows(duration: 118)
        precondition(longWindows.count == 15)
        let entireSections = longWindows.map { HighlightCandidate(span: $0, score: 4) }
        let digest = HighlightPlan.select(entireSections, duration: 118)
        precondition(!digest.isEmpty && digest.count <= 3)
        precondition(digest.allSatisfy { $0.duration <= 20 })
        precondition(digest.reduce(0, { $0 + $1.duration }) <= 118 * 0.35 + 0.000001)
        precondition(HighlightPlan.select(entireSections.reversed(), duration: 118) == digest)

        // Strong overlaps also cannot grow transitively from one end of a video to the other.
        let chain = stride(from: 0.0, through: 108.0, by: 4.0).map { event($0, $0 + 10) }
        let chainClips = HighlightPlan.select(chain, duration: 118)
        precondition(!chainClips.isEmpty && chainClips.allSatisfy { $0.duration <= 20 })
        precondition(chainClips.reduce(0, { $0 + $1.duration }) <= 118 * 0.35 + 0.000001)

        // Ranking must favor stronger events even when they occur later in the video.
        let ranked = HighlightPlan.select([event(5, 8), event(25, 28), event(50, 53),
                                          event(75, 78, 5)], duration: 118)
        precondition(ranked.count == 3 && ranked.contains(.init(start: 72, end: 81)))
        precondition(!ranked.contains(.init(start: 47, end: 56)))
        precondition(HighlightPlan.select([event(5, 8, 3)], duration: 118).isEmpty)
        precondition(HighlightPlan.select([event(0, 118, 5)], duration: 118).isEmpty)
        precondition(HighlightPlan.select([], duration: 118).isEmpty)

        // A short source can legitimately consist of a single event.
        precondition(HighlightPlan.select([event(0, 5, 5)], duration: 5) == [.init(start: 0, end: 5)])
        // Sweep dense candidates to check bounds, budgets, deterministic ordering and no overlap.
        for duration in [15.0, 45, 118, 600] {
            let events = stride(from: 0.0, to: duration - 2, by: 2.0).enumerated().map {
                event($0.element, min(duration, $0.element + 4), $0.offset % 2 + 4)
            }
            let clips = HighlightPlan.select(events, duration: duration)
            precondition(clips.count <= 3)
            precondition(clips.allSatisfy { $0.start >= 0 && $0.end <= duration && $0.duration > 0 && $0.duration <= 20 })
            precondition(clips.reduce(0, { $0 + $1.duration }) <= min(45, max(12, duration * 0.35)) + 0.000001)
            for (previous, next) in zip(clips, clips.dropFirst()) {
                precondition(previous.end <= next.start, "Separate clips must not duplicate footage")
            }
        }
        print("Highlight selection and timeline tests passed.")
    }
}
