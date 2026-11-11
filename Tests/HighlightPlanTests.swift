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
        let spans = try HighlightPlan.parse("{\"highlights\":[{\"start\":2,\"end\":9}]}", window: section)
        precondition(spans == [.init(start: 18, end: 25)], "Section-relative times need an absolute offset")
        let fenced = try HighlightPlan.parse("```json\n{\"highlights\":[]}\n```", window: section)
        precondition(fenced.isEmpty)
        for bad in ["hello", "{\"highlights\":[{\"start\":9,\"end\":2}]}",
                    "{\"highlights\":[{\"start\":-1,\"end\":2}]}",
                    "{\"highlights\":[{\"start\":0,\"end\":45}]}",
                    "{\"highlights\":[{\"start\":\"0:01\",\"end\":3}]}"] {
            do { _ = try HighlightPlan.parse(bad, window: section); preconditionFailure("Accepted malformed model output") }
            catch PlanError.invalidResponse { }
        }
        // A pass straddling windows must retain setup, flight, catch, and finish in one clip.
        let play = HighlightPlan.merge([.init(start: 9, end: 12), .init(start: 10, end: 18)], duration: 45)
        precondition(play == [.init(start: 6, end: 21)])
        let bounds = HighlightPlan.merge([.init(start: 0, end: 2), .init(start: 44, end: 45)], duration: 45)
        precondition(bounds == [.init(start: 0, end: 5), .init(start: 41, end: 45)])
        print("Highlight timeline tests passed.")
    }
}
