# ClipThing

A minimal SwiftUI app: choose a video, get Apple's suggested highlights, preview, save, or share. Analysis uses MediaIntelligence on device. No bundled third-party model, server, account, or API key.

## Run

1. Open `ClipThing.xcodeproj` in **Xcode 27** on a Mac.
2. Choose your signing team and bundle identifier.
3. Run on an iPhone/iPad with **iOS 27+**, required by HighlightAnalysisRequest.

When upgrading from SmolVLM2, use Product > Clean Build Folder and reinstall the app to clear old build resources. Old ignored `Models/` downloads may still exist locally; the project no longer references or bundles them.

## Behavior

- The Photos picker imports the selected video in its existing encoding. iCloud-only videos may need downloading. The provider file is copied before its callback returns.
- One whole-video HighlightAnalysisRequest supplies Apple's highlight ranges. The app validates, clamps, sorts, and removes exact duplicates. It does not pad, merge overlapping suggestions, rank them with an LLM, or impose a 20-second cap. Apple's selection quality is not guaranteed to recognize a complete sports play.
- A shared actor owns analysis across app windows. Cancellation keeps ownership until the outstanding request returns. Backgrounding cancels the job; export checks cancellation too.
- Clips export sequentially as temporary MOV files, using passthrough when supported. Save or share clips to keep them; importing another video removes previous temporary clips.
- Photos add-only permission is requested only when saving.

## Validation

Pure Swift tests cover original boundaries, overlapping and touching clips, invalid ranges, clamping, duplicates, and preservation of a complete play longer than 20 seconds. CI uses the Xcode 27 runner to compile the real Apple API and verify that no model weights or MLX resource bundle ship.

This Windows workspace cannot run Xcode or device analysis. Device testing must still establish highlight quality, latency, local/iCloud import, cancellation, portrait/HDR handling, and saving/sharing. No Vision or Foundation Models refinement has been added yet.

Reference: [Apple video highlights](https://developer.apple.com/documentation/mediaintelligence/finding-the-best-moments-in-a-video).
