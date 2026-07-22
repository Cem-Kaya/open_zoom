# Stability & Threading

Crash bugs, data races, and object-lifetime problems. This file contains the most urgent
work in the backlog: the known camera-switch crash from `TODO.md` has a diagnosed root
cause here (S1 + S2).

---

## S1. Camera-switch crash: CUDA surface destroyed while kernels are in flight

- **Status: DONE 2026-07-21.** `CudaInteropSurface` now has a `SynchronizeStream()`
  helper called from the destructor, the constructor failure path,
  `ReleaseDeviceBuffers()`, and `ReleaseTemporalHistory()`. The app additionally
  calls `presenter_->WaitForIdle()` before releasing the shared texture in
  `EnsureCudaSurface()` and the destructor. The transition-guard flag (fix step 2)
  was found unnecessary: `OnFrameTick`, `StartCameraCapture`, and
  `EnsureCudaSurface` all run on the Qt UI thread. Manual rapid camera-switch
  testing (fix step 3) still recommended on hardware.
- **Priority:** HIGH · **Effort:** medium · **Status:** Confirmed (destructor inspected; matches known bug in `TODO.md` line 3)
- **Evidence:** `src/cuda/cuda_interop.cpp:143` (`~CudaInteropSurface`), `src/app/app.cpp` `StartCameraCapture()` / `EnsureCudaSurface()` (~lines 1623–1714, 1839–1872)

**Problem.** When the user switches cameras, the app resets `cudaSurface_` (and the shared
texture) while the CUDA stream may still have kernels in flight referencing the surface
object, external memory, and device buffers. The destructor tears down in this order:
`cudaDestroySurfaceObject` → `cudaDestroyExternalMemory` → `cudaDestroyExternalSemaphore`
→ `cudaStreamDestroy` — with **no `cudaStreamSynchronize()` first**. Destroying the
external memory backing a surface that a queued kernel is about to read is a use-after-free
on the device. This matches the TODO.md entry "Switching cameras can trigger a crash
(likely during CUDA surface reinitialization)".

Note: `cudaStreamDestroy()` itself is documented as safe on a busy stream (it defers
cleanup), so the stream destruction is *not* the bug — the premature destruction of the
resources the queued work uses is.

**Fix.**
1. Add `if (stream_) cudaStreamSynchronize(stream_);` as the *first* statement of
   `~CudaInteropSurface()` (and of any `ReleaseDeviceBuffers()`-style teardown path that
   can run while the stream is live).
2. In `OpenZoomApp`, add a `cameraTransitioning_` flag (or mutex): set it at the top of
   `StartCameraCapture()`, have `OnFrameTick()` / `ProcessFrameWithCuda()` bail early
   while it is set, and clear it only after the new surface is fully initialized. This
   prevents a queued frame tick from racing the reset itself.
3. Manually test: rapidly switch cameras 20+ times with CUDA path active, with blur and
   temporal smoothing enabled (they use the device scratch buffers).

---

## S2. Media Foundation capture loop races `StopCapture()` / `StartCapture()`

- **Status: REFUTED 2026-07-21 — no change needed.** Verified: `running_` is
  `std::atomic<bool>`; `StopCapture()` joins the capture thread *before*
  resetting `sourceReader_`/`mediaSource_`, and those members are only written
  when no capture thread exists (`StartCapture` calls `StopCapture` first).
  `Flush()` from the control thread is the intended wakeup for a blocking
  synchronous `ReadSample`. Recorded in `verified-non-issues.md`.
- **Priority:** HIGH · **Effort:** medium · **Status:** Reported (verify exact lines before fixing)
- **Evidence:** `src/capture/media_capture.cpp` — `CaptureLoop()` reads `sourceReader_` (~line 376) without a lock; `StopCapture()` (~line 241) flushes and resets it from the main thread.

**Problem.** `sourceReader_` is copied into a local `ComPtr` in the capture thread without
synchronization while the main thread can `Flush()`/`Reset()` it during stop or camera
switch. The window between checking and copying the pointer is a race; a camera switch is
exactly when this fires — likely a second contributor to the TODO.md crash alongside S1.

**Fix.**
1. Add a `std::mutex sourceReaderMutex_`; hold it in `StopCapture()` around
   flush/reset and in `CaptureLoop()` when acquiring the local `ComPtr` copy.
2. Ensure `StartCapture()` (which already calls `StopCapture()` first) *joins* the
   capture thread before constructing the new reader, so the old loop can't observe the
   new reader mid-setup. An `std::atomic<bool> loopRunning_` plus `thread.join()` is
   enough; avoid detached threads here.

---

## S3. COM reference leak in `EnumerateCameras()`

- **Status: DONE 2026-07-21.** The redundant `AddRef()` was removed.
- **Priority:** MEDIUM · **Effort:** small (one-line fix) · **Status:** Confirmed
- **Evidence:** `src/capture/media_capture.cpp:110-111`; `include/openzoom/capture/media_capture.hpp:33`

**Problem.** `descriptor.activation` is a `Microsoft::WRL::ComPtr<IMFActivate>`
(media_capture.hpp:33). The assignment `descriptor.activation = devices[i];` already
AddRefs; the explicit `descriptor.activation->AddRef();` on the next line adds a second
reference that is never released. One `IMFActivate` leaks per device on **every**
enumeration (enumeration re-runs on refresh/camera change).

**Fix.** Delete line 111 (`descriptor.activation->AddRef();`). The subsequent
`devices[i]->Release()` loop and `CoTaskMemFree(devices)` are correct and stay.

---

## S4. Capture-thread vs UI-thread access to shared pipeline state

- **Priority:** HIGH · **Effort:** large · **Status:** Reported
- **Evidence:** `src/app/app.cpp` — camera callback writes `latestFrame_` under `cameraMutex_` (~lines 1852–1855); `OnFrameTick()` reads it under the same mutex (~1939–1942), but then processes via `cpuPipeline_` and reads tuning state (`zoomAmount_`, `blackWhiteThreshold_`, …) with no synchronization (~1948–1973).

**Problem.** The capture callback runs on a Media Foundation thread; UI slots mutate
tuning state on the Qt thread. Today the frame *data* handoff is locked but the pipeline
object and tuning parameters are not. Whether this is currently exploitable depends on
which thread `OnFrameTick` runs on (if it's a Qt timer, reads of tuning state are safe,
but `latestFrame_`'s buffer lifetime across the unlock is still suspect). Verify first.

**Fix (if confirmed).** Snapshot all tuning parameters into a plain `FrameSettings` struct
under one mutex at the top of `OnFrameTick()`, then process from the snapshot. Longer
term, move processing off the UI thread entirely: a `FrameProcessor` QObject on a worker
thread receiving frames via `Qt::QueuedConnection` and emitting processed results back.
Pairs naturally with the `PipelineOrchestrator` extraction in
[02-architecture-app-decomposition.md](02-architecture-app-decomposition.md).

---

## S5. Temporal-history buffers allocated/released without synchronization

- **Status: DONE 2026-07-21** (covered by the S1 fix — `ReleaseTemporalHistory()`
  and `ReleaseDeviceBuffers()` now synchronize the stream before `cudaFree`).
  Pre-allocating history at surface creation remains open as a minor optimization.
- **Priority:** MEDIUM · **Effort:** small · **Status:** Reported
- **Evidence:** `src/cuda/cuda_interop.cpp` — `EnsureTemporalHistory()` (~line 402), `ReleaseTemporalHistory()` (~line 422)

**Problem.** `ReleaseTemporalHistory()` can run during camera switch while
`ProcessFrame()` is using the history buffers on the stream — same shape as S1 (freeing
device memory with dependent work queued). Fixing S1's stream-sync-before-teardown likely
covers this; confirm `ReleaseTemporalHistory()` also syncs the stream before
`cudaFree`, and that history state changes only happen from one thread.

**Fix.** Route all temporal-history teardown through one method that synchronizes
`stream_` first. Prefer pre-allocating history at surface creation (dimensions are known)
over on-demand allocation inside `ProcessFrame()`.

---

## S6. D3D12↔CUDA fence protocol is implicit and unrecoverable on failure

- **Partially addressed 2026-07-21:** `ProcessFrameWithCuda()` now re-seeds
  `sharedFenceCounter_` from `presenter_->GetLastSignaledFenceValue()` every
  CUDA frame, so fence values stay monotonic when CPU-path (debug view) frames
  interleave — previously the fence could be signaled backwards. Still open:
  documenting the protocol in one place and a resync path after N consecutive
  ProcessFrame failures.
- **Priority:** MEDIUM · **Effort:** small–medium · **Status:** Reported
- **Evidence:** `src/app/app.cpp` — `ResetCudaFenceState()` (~1615–1621), fence-value choreography (~1767–1807); `src/d3d12/presenter.cpp` `PresentFromTexture()` (~line 163); `src/cuda/cuda_interop.cpp` `ProcessFrame()` semaphore wait/signal (~line 501)

**Problem.** Fence values are sequenced by `sharedFenceCounter_` in app.cpp, but the
wait/signal contract between the D3D12 side and the CUDA side is undocumented and split
across three files. If `ProcessFrame()` fails after the counter was advanced, the signal
never happens and the other side can wait forever; there is no resync path.

**Fix.**
1. Document the protocol in one place (header comment or a small `FenceSyncController`
   struct owning the counter): who signals, who waits, value progression per frame.
2. On CUDA processing failure, either roll the counter back or force a
   `ResetCudaFenceState()` resync after N consecutive failures, so one bad frame cannot
   wedge the pipeline.

---

## S7. Mid-stream camera format changes are never detected

- **Status: DONE 2026-07-22.** `CaptureLoop()` now handles
  `MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED`, re-reads the active format and
  rejects unsupported changes before delivering another frame.
- **Priority:** MEDIUM · **Effort:** medium · **Status:** Reported
- **Evidence:** `src/capture/media_capture.cpp` — `CaptureLoop()` caches the negotiated format once (~line 384) and never re-queries.

**Problem.** Some camera drivers change stride/resolution mid-stream (auto-exposure mode
switches, USB renegotiation). A stale cached format means every downstream stride
calculation is wrong → corrupted frames or out-of-bounds reads in the converters
(see also V2 in [05-robustness-validation.md](05-robustness-validation.md)).

**Fix.** `IMFSourceReader::ReadSample` reports `MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED`
in its stream-flags output parameter — check it on every sample (this is the idiomatic MF
approach; cheaper and more correct than periodic re-query). On change, re-query
`GetCurrentMediaType`, update the cached format, and notify the pipeline to reset
(temporal history, CUDA surface dimensions).
