import AVKit
import PhotosUI
import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var model = HighlightModel()
    @State private var selection: PhotosPickerItem?
    @State private var preview: HighlightClip?

    var body: some View {
        // PhotosPicker's label can be Sendable. Capture a value, not the main-actor model.
        let pickerTitle = model.clips.isEmpty ? "Choose Video" : "Choose Another Video"
        return NavigationStack {
            VStack(spacing: 20) {
                PhotosPicker(selection: $selection, matching: .videos, preferredItemEncoding: .current) {
                    Label(pickerTitle, systemImage: "video.badge.plus")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isBusy)

                Text(model.status)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if model.isBusy {
                    ProgressView()
                    // Saving to Photos is an atomic operation; only offer cancellation during analysis/export.
                    if model.canCancel {
                        Button("Cancel", role: .cancel) { model.cancel() }
                    }
                }

                if !model.clips.isEmpty {
                    List(Array(model.clips.enumerated()), id: \.element.id) { index, clip in
                        HStack(spacing: 12) {
                            Button {
                                preview = clip
                            } label: {
                                HStack(spacing: 12) {
                                    Image(systemName: "play.circle.fill").font(.largeTitle)
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Highlight \(index + 1)").font(.headline)
                                        Text("\(Int(clip.duration.rounded())) sec · from \(timestamp(clip.start))")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .buttonStyle(.plain)
                            Spacer()
                            ShareLink(item: clip.url) { Image(systemName: "square.and.arrow.up") }
                                .accessibilityLabel("Share highlight \(index + 1)")
                            Button { model.save([clip]) } label: { Image(systemName: "square.and.arrow.down") }
                                .buttonStyle(.borderless)
                                .accessibilityLabel("Save highlight \(index + 1)")
                        }
                        .padding(.vertical, 8)
                    }
                    .listStyle(.plain)
                    .disabled(model.isBusy)

                    Button("Save All to Photos") { model.save(model.clips) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isBusy)
                } else {
                    Spacer()
                    Image(systemName: "sparkles.tv").font(.system(size: 64)).foregroundStyle(.tint)
                        .accessibilityHidden(true)
                    Text("Just the good parts.").font(.title2.bold())
                    Text("Video analysis stays on your device.")
                        .font(.subheadline).foregroundStyle(.secondary)
                    Spacer()
                }
            }
            .padding()
            .navigationTitle("ClipThing")
            .onChange(of: selection) { _, item in
                guard let item else { return }
                model.process(item)
            }
            .onChange(of: model.isImporting) { _, importing in
                if !importing { selection = nil }
            }
            .onChange(of: scenePhase) { _, phase in
                // Cancel analysis when backgrounded; retain the job gate until it actually stops.
                if phase == .background { model.cancel() }
            }
            .sheet(item: $preview) { ClipPreview(clip: $0) }
            .alert("Couldn't finish", isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) { model.errorMessage = nil }
            } message: { Text(model.errorMessage ?? "") }
        }
    }

    private func timestamp(_ seconds: Double) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

private struct ClipPreview: View {
    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer

    init(clip: HighlightClip) {
        _player = State(initialValue: AVPlayer(url: clip.url))
    }

    var body: some View {
        NavigationStack {
            VideoPlayer(player: player)
                .navigationTitle("Preview")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { Button("Done") { dismiss() } }
                .onAppear { player.play() }
                .onDisappear { player.pause(); player.replaceCurrentItem(with: nil) }
        }
    }
}
