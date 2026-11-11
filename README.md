# ClipThing

A minimal SwiftUI iPhone/iPad app: choose one video, automatically find highlights, preview, share, or save them to Photos. No server, account, packages, or API key.

## Run

1. On a Mac, open `ClipThing.xcodeproj` in Xcode with the iOS 27 SDK and MediaIntelligence framework.
2. Select the ClipThing target, choose your signing team, and change the example bundle identifier to your own.
3. Run on a compatible device running iOS 27 or later. Real-device validation is necessary for on-device media analysis.

Apple documents `VideoAnalyzer` and `HighlightAnalysisRequest` as iOS 27+ APIs. Older SDKs cannot compile this app. Hardware support and highlight quality need validation on your device.

## Behavior

- The system Photos picker imports only your selected video. iCloud-only videos may require a download.
- Apple's Media Intelligence selects highlight time ranges locally. Overlapping ranges are merged and clipped to the source duration.
- Clips export sequentially to temporary MOV files, using passthrough when supported to avoid unnecessary encoding. Preview/save/share become available when processing finishes.
- One operation owns the pipeline at a time. Cancellation keeps the pipeline locked until work stops; saving to Photos is an atomic operation without a cancel button.
- Choosing another video deletes the previous temporary clips. Save clips you want to keep. Temporary files are not a permanent library; the OS may purge them.
- Photos add-only access is requested only when saving. Sharing doesn't need that permission.

## Validation

Created on Windows; no Xcode/iOS SDK is available here. Swift syntax can be parsed locally, but this is not an iOS build or runtime test.

On a real device, verify: local and iCloud import; portrait/HDR video and audio preservation; highlight quality; cancellation during import/analysis/export followed by another import; no-highlight results; denied Photos access; individual save, Save All, and Share. Check both short and long videos. Apple chooses the number and duration of highlights; no quality or processing-speed guarantee is assumed.

References: [Media Intelligence](https://developer.apple.com/documentation/mediaintelligence), [highlight analysis](https://developer.apple.com/documentation/mediaintelligence/finding-the-best-moments-in-a-video), [availability](https://developer.apple.com/documentation/mediaintelligence/videoanalyzer.md).
