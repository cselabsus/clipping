# ClipThing

A minimal SwiftUI app: choose a video, get suggested highlights, preview, save, or share. Analysis uses **SmolVLM2 2.2B, 4-bit**, bundled inside the app. No server, account, or API key.

## Run

1. On a Mac, open `ClipThing.xcodeproj` in **Xcode 26**.
2. Choose your signing team and change the example bundle identifier to your own.
3. Run on a real iPhone/iPad running **iOS 26+** with at least **6 GB physical RAM**. This is a conservative app guard, not a validated compatibility list. Memory pressure can still affect inference.
4. The first build's **Prepare Bundled Model** step downloads approximately **1.46 GB** of weights plus configuration/tokenizer files. Allow time and disk space for the model, build products, and installed app. Subsequent builds reuse and verify these files.

No runtime model download, inference API, or video upload exists. The developer's first build needs internet; iCloud-only videos may also need downloading by Photos. Fully local videos can be analyzed offline. MLX inference cannot run in the iOS simulator.

Model preparation uses Python 3 and curl. To prepare or verify manually:

```sh
python3 Scripts/prepare_model.py
python3 Scripts/prepare_model.py --verify-only
```

Weights are excluded from Git. `Scripts/model-manifest.json` pins the model revision, byte sizes, and SHA-256 hashes. Downloads are serialized, verified, and atomically installed. A failed checksum stops the build. Do not commit model weights to GitHub.

## Behavior

- The system Photos picker imports only your selected video. iCloud-only videos may require a download.
- Import requests the existing encoding and accepts QuickTime, MP4, and generic video file representations. The picker selection is retained until import finishes, and the temporary provider file is copied before its callback returns. Import failures include content-type diagnostics; no whole-video in-memory fallback is used.
- Videos up to 10 minutes are processed in 12-second windows with 4 seconds of overlap, at 2 frames/second (at most 24 frames per inference).
- The video processor preserves orientation, converts to sRGB, resizes to 384x384, normalizes to [-1, 1], and attaches section-relative timestamps. It derives **81 image tokens per frame** from this checkpoint's geometry `(384 / 14 / 3)^2` using integer division. The library's default processor assumes 64, so this app prepares its own timestamped video inputs and validates token counts.
- The prompt asks for complete events, including setup/action/outcome and throw/flight/catch/finish in sports. JSON timestamps are strictly validated, offset to the original video, padded by 3 seconds, and merged across overlapping windows.
- Malformed model output fails the job. An empty valid response means no highlights were selected. No fallback to Apple's scorer or arbitrary clips exists.
- This is sampled visual analysis, not audio analysis or ball tracking. Small/fast objects can still be missed. Valid timestamps do not establish that a highlight is correct.
- Clips export sequentially to temporary MOV files, using passthrough when supported to avoid unnecessary encoding. Preview/save/share become available when processing finishes.
- A shared actor owns inference across app windows. The model is scoped to one job and released before export; the MLX reuse cache is capped at 32 MB. Generation is bounded to 180 tokens at temperature zero, checks cancellation per token, and synchronizes GPU work before returning. Cancellation during loading or vision prefill may take time; the pipeline stays locked until work stops. Backgrounding cancels analysis. Saving to Photos is atomic.
- Choosing another video deletes the previous temporary clips. Save clips you want to keep. Temporary files are not a permanent library; the OS may purge them.
- Photos add-only access is requested only when saving. Sharing doesn't need that permission.

## Validation

The GitHub Actions `iOS build` workflow compiles the app on macOS against the iPhone SDK and checks model bundling. Pure Swift tests cover window coverage, timestamp offsets, invalid outputs, bounds, and merging a play across windows. Python tests check corrupt downloads and atomic installation.

This Windows workspace cannot execute MLX or Xcode. Device inference and quality still need verification. Test the 45-second touchdown video and confirm the exported clip includes throw, flight, catch, and finish. Also test local/iCloud import, airplane mode with local media, portrait/HDR videos, cancellation followed by another import, denied Photos access, saving/sharing, and long-video memory usage. A successful build does not establish model accuracy.

## Pinned components

- [SmolVLM2 2.2B 4-bit MLX](https://huggingface.co/smdesai/SmolVLM2-2.2B-Instruct-4bit), revision `0a4ab666f99095e6c6bd0e1e49aca8822fc46df2` (Apache-2.0).
- [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm/tree/2.29.3) 2.29.3; MLX Swift 0.29.1; Swift Transformers 1.1.6.
- License texts and redistribution notices are in `ThirdPartyNotices.txt`, also bundled in the app.
