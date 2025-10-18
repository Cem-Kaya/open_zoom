# Rotation UI + Pipeline Notes

- The Windows build helper script was launching the stale `build/Release/open_zoom.exe`. The
  actual MSBuild target drops binaries in `build/cmake/Release`. Update
  `scripts/build_and_run.bat` to probe that folder first so new UI changes load immediately.
- `MainWindow` now exposes a dedicated `Rotation:` combo box placed in its own row above the
  camera selector (`src/ui/main_window.cpp`). The control wires to
  `OpenZoomApp::OnRotationSelectionChanged`, which rotates the capture before CPU/GPU processing
  and keeps the zoom centre aligned.
- For 90°/270° orientations we resample the rotated frame to the widget aspect before handing it
  to CUDA, so the GPU path keeps running without introducing black bars or stretching. The resize
  happens in `CpuFramePipeline::ResampleToFill`.
- If the UI ever looks stale after rebuilding, confirm the helper is executing
  `build/cmake/Release/open_zoom.exe` and not an older copy in `build/Release`.
