import AVFoundation
import CoreTransferable
import OSLog
import Photos
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct ImportedVideo: Transferable, Sendable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .quickTimeMovie) { received in
            try copyVideo(received.file)
        }
        FileRepresentation(importedContentType: .mpeg4Movie) { received in
            try copyVideo(received.file)
        }
        FileRepresentation(importedContentType: .movie) { received in
            try copyVideo(received.file)
        }
        FileRepresentation(importedContentType: .video) { received in
            try copyVideo(received.file)
        }
        FileRepresentation(importedContentType: .audiovisualContent) { received in
            try copyVideo(received.file)
        }
    }

    private static func copyVideo(_ source: URL) throws -> ImportedVideo {
        try Task.checkCancellation()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ClipThing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent("source")
            .appendingPathExtension(source.pathExtension)
        do {
            // The provider's URL is temporary: copy it before returning from its callback.
            try FileManager.default.copyItem(at: source, to: destination)
            try Task.checkCancellation()
            return ImportedVideo(url: destination)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }
}

struct HighlightClip: Identifiable {
    let id = UUID()
    let url: URL
    let start: Double
    let duration: Double
}

@MainActor
final class HighlightModel: ObservableObject {
    @Published private(set) var clips: [HighlightClip] = []
    @Published private(set) var isBusy = false
    @Published private(set) var canCancel = false
    @Published private(set) var isImporting = false
    @Published private(set) var status = "Choose a video. Get its best moments."
    @Published var errorMessage: String?

    private var job: Task<Void, Never>?
    private var workingDirectory: URL?
    private let logger = Logger(subsystem: "ClipThing", category: "VideoImport")

    func process(_ selection: PhotosPickerItem) {
        guard !isBusy else { return }
        cleanUp()
        errorMessage = nil
        isBusy = true
        canCancel = true
        isImporting = true
        status = "Loading video…"
        job = Task {
            defer { isImporting = false; isBusy = false; canCancel = false; job = nil }
            do {
                let video: ImportedVideo
                do {
                    guard let imported = try await selection.loadTransferable(type: ImportedVideo.self) else {
                        throw ClipError.importFailed
                    }
                    video = imported
                } catch {
                    if Task.isCancelled || error is CancellationError { throw CancellationError() }
                    let types = selection.supportedContentTypes.map(\.identifier).joined(separator: ", ")
                    let failure = error as NSError
                    logger.error("Import failed. Types: \(types, privacy: .public). Error: \(failure.domain, privacy: .public) (\(failure.code)): \(failure.localizedDescription, privacy: .private)")
                    throw ClipError.transferFailed(details: "\(failure.domain) (\(failure.code)); types: \(types)")
                }
                workingDirectory = video.url.deletingLastPathComponent()
                isImporting = false
                try Task.checkCancellation()
                let source = AVURLAsset(url: video.url)
                let videoTracks = try await source.loadTracks(withMediaType: .video)
                try Task.checkCancellation()
                guard !videoTracks.isEmpty else {
                    throw ClipError.importFailed
                }
                let duration = try await source.load(.duration)
                let highlights = try await AppleHighlightEngine.shared.highlights(in: video.url, duration: duration.seconds) {
                    [weak self] message in
                    await self?.updateAnalysisStatus(message)
                }
                try Task.checkCancellation()
                let bounds = CMTimeRange(start: .zero, duration: duration)
                // Clamp only: selection already owns event boundaries and duplicate removal.
                // Re-merging here would undo the selector and join separate highlights.
                let ranges = Self.normalized(highlights.map {
                    CMTimeRange(start: CMTime(seconds: $0.start, preferredTimescale: 600),
                                end: CMTime(seconds: $0.end, preferredTimescale: 600))
                }, within: bounds)
                guard !ranges.isEmpty else {
                    status = "No highlights found. Try another video."
                    cleanUp()
                    return
                }
                for (index, range) in ranges.enumerated() {
                    try Task.checkCancellation()
                    status = "Preparing clip \(index + 1) of \(ranges.count)…"
                    let url = video.url.deletingLastPathComponent()
                        .appendingPathComponent("Highlight-\(index + 1).mov")
                    // Passthrough avoids re-encoding when the source supports it.
                    let presets = AVAssetExportSession.exportPresets(compatibleWith: source)
                    let preset = presets.contains(AVAssetExportPresetPassthrough)
                        ? AVAssetExportPresetPassthrough : AVAssetExportPresetHighestQuality
                    guard let exporter = AVAssetExportSession(asset: source, presetName: preset) else {
                        throw ClipError.exportFailed
                    }
                    exporter.timeRange = range
                    try await exporter.export(to: url, as: .mov)
                    try Task.checkCancellation()
                    clips.append(HighlightClip(url: url, start: range.start.seconds, duration: range.duration.seconds))
                }
                try? FileManager.default.removeItem(at: video.url)
                status = "\(clips.count) highlight\(clips.count == 1 ? "" : "s") ready"
            } catch {
                let cancelled = Task.isCancelled || error is CancellationError
                cleanUp()
                status = cancelled ? "Cancelled. Choose a video to start again." : "Try another video."
                if !cancelled { errorMessage = error.localizedDescription }
            }
        }
    }

    func cancel() {
        guard canCancel, job != nil else { return }
        canCancel = false
        status = "Cancelling…"
        job?.cancel()
        // Keep the gate closed until the underlying operation has actually stopped.
    }

    private func updateAnalysisStatus(_ message: String) {
        guard canCancel else { return } // Ignore progress after cancellation has been requested.
        status = message
    }

    func save(_ selected: [HighlightClip]) {
        guard !isBusy, !selected.isEmpty else { return }
        isBusy = true
        status = "Saving to Photos…"
        job = Task {
            defer { isBusy = false; job = nil }
            do {
                let permission = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                guard permission == .authorized || permission == .limited else {
                    throw ClipError.photoPermission
                }
                try Task.checkCancellation()
                let urls = selected.map(\.url)
                try await PHPhotoLibrary.shared().performChanges {
                    for url in urls {
                        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                    }
                }
                status = "Saved to Photos"
            } catch {
                status = "Your clips are ready"
                if !Task.isCancelled { errorMessage = error.localizedDescription }
            }
        }
    }

    private func cleanUp() {
        clips = []
        if let workingDirectory { try? FileManager.default.removeItem(at: workingDirectory) }
        workingDirectory = nil
    }

    private static func normalized(_ ranges: [CMTimeRange], within bounds: CMTimeRange) -> [CMTimeRange] {
        ranges.filter { $0.isValid && !$0.isIndefinite }
            .map { CMTimeRangeGetIntersection($0, otherRange: bounds) }
            .filter { $0.start.seconds.isFinite && $0.duration.seconds.isFinite && $0.duration.seconds > 0 }
            .sorted { $0.start < $1.start }
    }
}

private enum ClipError: LocalizedError {
    case importFailed, exportFailed, photoPermission
    case transferFailed(details: String)
    var errorDescription: String? {
        switch self {
        case .importFailed: "This video couldn't be loaded. If it's in iCloud, check your connection and try again."
        case .transferFailed(let details): "Photos couldn't provide this video. Open it in Photos and let it download fully, then try again.\n\nImport details: \(details)"
        case .exportFailed: "This video couldn't be exported. Try a different video."
        case .photoPermission: "Allow ClipThing to add photos in Settings, or use Share to save your clips elsewhere."
        }
    }
}
