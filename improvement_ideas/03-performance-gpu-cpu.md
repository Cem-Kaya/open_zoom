# Performance: GPU & CPU Pipeline

For a real-time magnifier, latency *is* the product. P1–P3 (present/upload/readback
pipelining) are done; the single biggest remaining win is P11 (pinned-host upload
staging). Measure before/after each change — add the frame-timing instrumentation (P8)
first so improvements are quantified, not assumed.

---

## P1. Remove the per-frame full-GPU stall in the present path

- **Status: DONE 2026-07-21.** Both present paths are now pipelined with
  per-back-buffer command allocators and per-slot fence pacing; full drains
  remain only for resize, teardown (`WaitForIdle()`), readback, and the
  no-external-semaphore CUDA fallback (which has no other cross-API sync).
  App-side, the shared fence counter is re-seeded each CUDA frame to stay
  monotonic across path switches, and the queue is drained before the shared
  texture is released.
- **Priority:** HIGH · **Effort:** medium · **Status:** Confirmed
- **Evidence:** `src/d3d12/presenter.cpp:147` — `Present()` ends with an unconditional `WaitForGpu()` right after `swapChain_->Present(1, 0)`. `PresentFromTexture()` and `ReadbackTexture()` (~line 498) have the same pattern.

**Problem.** Every frame, the CPU blocks until the GPU fully drains. This serializes
CPU upload, GPU copy, and present — throughput is capped at `1 / (upload + GPU + present)`
instead of pipelining, and input-to-photon latency grows by a full GPU frame.

**Fix.** Standard frame pipelining:
1. Per-back-buffer command allocators and an in-flight fence value per buffer
   (the swap chain already has N back buffers).
2. In `Present()`, signal the fence after `ExecuteCommandLists` and return. At the *top*
   of the next `Present()`, wait only until the fence for the back buffer about to be
   reused has passed.
3. Keep a full `WaitForGpu()` only for resize/teardown paths.

Interacts with P2 (the upload buffer must also be per-frame or ring-allocated, or the CPU
write would race the in-flight copy). Do P1+P2 together.

---

## P2. Ring-buffer the upload heap

- **Status: DONE 2026-07-21** (implemented together with P1: one persistently
  mapped upload buffer per frame slot).
- **Priority:** MEDIUM · **Effort:** medium · **Status:** Confirmed (single persistent mapped upload buffer at `EnsureUploadBuffer`, ~presenter.cpp:406)
- **Evidence:** `src/d3d12/presenter.cpp` — one `uploadBuffer_`, CPU-written every frame

**Problem.** With P1 in place, a single upload buffer would be overwritten by frame N+1
while frame N's copy is still in flight. Today this is masked by the P1 stall.

**Fix.** N upload buffers (N = back-buffer count), indexed by frame; the fence wait in
P1 step 2 guarantees safety before reuse.

---

## P3. Asynchronous readback for recording/snapshots

- **Status: DONE 2026-07-22.** Recording and periodic assistive grabs use a
  two-slot asynchronous readback ring with per-slot command allocators and
  fence completion polling. Photo capture and explicit on-demand analysis keep
  the synchronous path where immediate output is required.
- **Priority:** MEDIUM · **Effort:** medium · **Status:** Confirmed (`ReadbackTexture()` calls `WaitForGpu()`, presenter.cpp ~413–515)
- **Evidence:** `src/d3d12/presenter.cpp:413-515`; callers in `src/app/app.cpp` (~400, ~1983, ~2191)

**Problem.** Every recorded frame and every assistive-analysis frame does a synchronous
readback with a full GPU stall on the render path. At 30 fps recording this doubles the
stall count per frame (present + readback).

**Fix.** Double-buffered readback: issue the copy-to-readback with a fence signal, return
immediately, and have the *next* readback call collect the previous frame's result once
its fence passed (one frame of latency in recordings is unnoticeable). Recording and
assistive analysis tolerate latency; snapshots can keep the synchronous path.

---

## P4. Shared-memory tiling for the CUDA Gaussian blur

- **Priority:** MEDIUM · **Effort:** large · **Status:** CLOSED-WONTFIX 2026-07-23 for the prescribed three-box implementation
- **Evidence:** `src/cuda/cuda_kernels.cu` — `GaussianBlurHorizontalKernel` / `GaussianBlurVerticalKernel` read neighborhoods straight from global memory; radius can be up to 50.

**Problem.** Each output pixel reads up to 101 global-memory samples with no staging;
adjacent threads re-read overlapping windows. At 1080p with a large radius this is the
most expensive kernel in the pipeline by far.

**Fix.** Cooperative tile loading into `__shared__` memory (block width + 2×radius),
`__syncthreads()`, then filter from shared memory. Alternative with better
effort/benefit ratio for very large radii: replace with 2–3 iterated box blurs
(O(1) per pixel via running sum), which converges to Gaussian and makes radius cost-free.
Benchmark both against the current kernel before committing.

**Batch-B gate result.** The 1920x1080 radius-25, sigma-8.333 running-sum
prototype took 4.159 ms versus 0.355 ms for the exact Gaussian on the RTX
4090 Laptop GPU (11.7x slower). It was removed before merge, so the exact
visual path remains.

---

## P5. Cache Gaussian kernel weights; avoid re-upload on slider moves

- **Status: RESOLVED (verified already implemented) 2026-07-22.**
  `CudaInteropSurface::EnsureGaussianKernel()` (`src/cuda/cuda_interop.cpp`,
  ~1128) caches `cachedKernelRadius_` / `cachedKernelSigma_` and skips
  `UploadGaussianKernel()` when the pair is unchanged. No re-upload happens on
  steady-state frames. The remaining inefficiency — the upload itself being
  synchronous when it *does* run — is tracked separately as P13.
- **Priority:** MEDIUM · **Effort:** small–medium · **Status:** Reported
- **Evidence:** `src/cuda/cuda_kernels.cu` — `UploadGaussianKernel()` recomputes weights and calls synchronous `cudaMemcpyToSymbol` whenever radius/sigma changes.

---

## P6. Device-buffer reallocation churn on resolution change

- **Priority:** LOW–MEDIUM · **Effort:** medium · **Status:** Deferred by Batch B 2026-07-23 (not trivial)
- **Evidence:** `src/cuda/cuda_interop.cpp` — `EnsureDeviceBuffers()` frees and reallocates buffers A/B/scratch whenever dimensions change.

**Problem.** Resolution changes (mode switch, camera switch) free + `cudaMallocPitch`
three buffers mid-pipeline. Not per-frame today (caching exists ~312–316), so this is
lower priority than the agents' original report suggested.

**Fix.** Allocate at max-supported resolution once per surface lifetime, or use a CUDA
memory pool (`cudaMallocAsync`, CUDA 11.2+). Only worth doing after P1–P3.

**Batch-B decision.** Resolution changes recreate the owning D3D12 interop
surface, so cross-surface allocation reuse requires a broader pool/lifetime
design and did not satisfy the handoff's "only if trivial" condition.

---

## P7. CPU pipeline: per-frame allocations and full-frame copies

- **Priority:** LOW (downgraded 2026-07-22) · **Effort:** medium · **Status:** Reported
- **Scope note 2026-07-22:** the CPU effects path is deprecated and NV12/YUY2
  frames now convert/rotate on the GPU, so this code only runs for the debug
  composite view, other camera formats, GPU-unavailable passthrough, and
  per-frame fallback. The stage copies still exist (`stageBw_ = stageRaw_` at
  `frame_pipeline.cpp` ~195) but no longer sit on the main processing path.
- **Evidence:** `src/common/frame_pipeline.cpp` — `BuildStages()` (~150–158) copies full frames between stage buffers (`stageBw_ = stageRaw_`, etc.); `src/common/image_processing.cpp` `resize()` calls in converters (~34, 49, 85, 144)

**Problem.** Stage buffers are copied even when a stage is disabled; `resize()` churns
when dimensions fluctuate. On the CPU fallback path (the guaranteed-available path for
users without NVIDIA GPUs) this costs real frame budget.

**Fix.**
1. Only materialize intermediate stage copies when Debug View is enabled; otherwise
   process in place or ping-pong between two persistent buffers.
2. `reserve()` stage buffers to the first frame's size; never shrink.
3. Consider row-parallelism via `QtConcurrent::map` or a simple thread pool for the
   convert/zoom/blur loops — these are embarrassingly parallel.

---

## P8. Frame-timing instrumentation and graceful degradation

- **Priority:** MEDIUM · **Effort:** medium · **Status:** Confirmed (re-verified 2026-07-22 — still no frame/stage timing anywhere; the only `cudaEventRecord` use is keystone snapshot synchronization, not timing)
- **Evidence:** absence of any timing in `src/app/app.cpp` `OnFrameTick()` and `src/cuda/cuda_interop.cpp` `ProcessFrame()`

**Problem.** There is no measurement of frame latency or per-stage cost, so regressions
are invisible and there is no load-shedding: when the GPU falls behind, work queues up
and the magnifier lags — the worst failure mode for a low-vision user tracking text.

**Fix.**
1. CPU-side: wall-clock per `OnFrameTick`, rolling average, surfaced in the status label
   (e.g. "34 ms/frame · GPU path").
2. GPU-side: `cudaEventRecord`/`cudaEventElapsedTime` around the kernel pipeline in
   debug builds.
3. Degradation policy: if frame time exceeds budget (e.g. >50 ms for 3 consecutive
   frames), temporarily disable the most expensive optional stages (sharpen, blur) and
   tell the user via the status label.

---

## P9. Kernel launch-config tuning (block size)

- **Priority:** LOW · **Effort:** small · **Status:** Reported
- **Evidence:** all kernels launch with fixed `dim3(16, 16)` blocks (`src/cuda/cuda_kernels.cu`)

**Fix.** Try 32×8 (same 256 threads, better coalescing for row-major access) and use
`cudaOccupancyMaxPotentialBlockSize` per kernel at init. Low expected gain; do last and
only with the P8 instrumentation in place to confirm it helps.

---

## P10. Verify no dead/unsafe in-place zoom kernel remains

- **Priority:** LOW · **Effort:** small · **Status:** Reported (needs verification)
- **Evidence:** `src/cuda/cuda_kernels.cu` — surface-object `ZoomKernel` (~183) reads and writes the same surface in-place; a separate `ZoomLinearKernel` (~275) uses distinct src/dst buffers.

**Fix.** Check which launcher `ProcessFrame()` actually uses. If only the linear variant
is live, delete the surface-object variant (dead code with a latent race). If the
in-place variant is live, switch to the linear one.

---

## P11. Pinned-host staging ring for frame uploads

- **Priority:** HIGH · **Effort:** medium · **Status:** DONE 2026-07-23
- **Evidence:** `src/cuda/cuda_interop.cpp` `ProcessFrame()` (~1278–1325) issues
  `cudaMemcpy2DAsync` host→device from pageable memory: the BGRA path uploads from
  `CpuFramePipeline`'s `stageRaw_` (`std::vector`), and the raw NV12/YUY2 path uploads
  plane pointers from the capture-side buffer. None of these are pinned
  (no `cudaHostAlloc` / `cudaHostRegister` anywhere in the repo).

**Problem.** `cudaMemcpy2DAsync` from *pageable* host memory is effectively synchronous:
the driver stages through an internal pinned bounce buffer and the call blocks until the
staging copy completes, so the upload cannot overlap kernel execution or the previous
frame's GPU work. The P1/P2 present-path pipelining does not help the upload side.

**Fix.** A small ring (2–3 slots) of pinned staging buffers (`cudaHostAlloc`, or
`cudaHostRegister` on persistent buffers), sized for the largest plane set. CPU copies
the camera frame into the current slot, `cudaMemcpy2DAsync` reads from pinned memory
(truly asynchronous on `stream_`), and a per-slot event/fence guards reuse. Applies to
both the BGRA and the raw NV12/YUY2 upload paths.

**Implemented.** Two `cudaMallocHost` slots are row-packed for all formats;
per-slot upload events guard reuse. Capture-to-app ownership moves instead of
copying twice. A representative 1080p handoff/upload microbenchmark improved
from 1.307 to 0.841 ms (five-run mean, 35.7%).

---

## P12. Dead constant: `gGaussianKernelSize` uploaded but never read

- **Priority:** LOW · **Effort:** trivial · **Status:** DONE 2026-07-23
- **Original evidence:** `src/cuda/cuda_kernels.cu` declared
  `__constant__ int gGaussianKernelSize;` and `UploadGaussianKernel()` uploaded
  it via `cudaMemcpyToSymbol`; no kernel read it (the blur kernels use
  `gGaussianRadius`). Both dead references are now removed.

**Fix.** Delete the declaration and the upload — one fewer synchronous symbol copy per
kernel change.

---

## P13. `UploadGaussianKernel()` is fully synchronous (device-wide sync)

- **Priority:** LOW–MEDIUM · **Effort:** small · **Status:** DONE 2026-07-23
- **Evidence:** `src/cuda/cuda_kernels.cu` `UploadGaussianKernel()` (~1412–1455) uses
  default-stream `cudaMemcpyToSymbol` three times and ends with a full
  `cudaDeviceSynchronize()`, despite receiving a `cudaStream_t stream` parameter it
  never uses.

**Problem.** Every blur radius/sigma change stalls the whole device. Thanks to the P5
cache this only fires on slider moves, but a slider drag emits many changes and each one
hitches the live view.

**Fix.** Use `cudaMemcpyToSymbolAsync(..., stream)` for all three symbols and drop the
`cudaDeviceSynchronize()` — the subsequent kernel launches on the same stream already
order correctly after the copies.

**Measured.** Radius/sigma upload host time improved from 0.064 to 0.025 ms
(five-run mean, 61.7%) after the dead copy and device sync were removed.

---

## P14. Stabilization projections use per-pixel global atomics

- **Priority:** MEDIUM · **Effort:** small–medium · **Status:** DONE 2026-07-23
- **Evidence:** `src/cuda/cuda_kernels.cu` (~735–747) — the projection kernel does
  `atomicAdd(&colProj[x], value)` and `atomicAdd(&rowProj[y], value)` for every pixel of
  the downsampled luma image, straight into global memory.

**Problem.** Every thread issues two global atomic adds; threads in the same
block/column contend heavily on the same accumulator words. (The in-source comment
claims contention is negligible; profiling says otherwise for warp-aligned columns.)
The auto-contrast histogram kernel (~1137) already shows the right pattern: accumulate
into `__shared__` bins, then merge once per block.

**Fix.** Same treatment: per-block `__shared__` partial row/column sums, `__syncthreads()`,
then one global `atomicAdd` per bin per block — cuts global atomics by ~two orders of
magnitude.

**Measured.** The 320x180 projection stage improved from 0.039 to 0.037 ms
(five-run mean, 5.9%) with 0.0 maximum absolute error against CPU row/column
sums.

---

## P15. Use texture objects for NIS/FSR (and other bilinear-sampling kernels)

- **Priority:** LOW–MEDIUM · **Effort:** medium · **Status:** Confirmed 2026-07-22 (no `cudaTextureObject_t` / `tex2D` anywhere in `cuda_kernels.cu`)
- **Evidence:** the NIS and FSR EASU/RCAS kernels sample multi-tap neighborhoods with
  hand-rolled bilinear interpolation from raw global-memory pointers.

**Problem.** These upscalers are the most sample-hungry kernels in the pipeline; raw
global loads skip the texture cache and pay for manual address clamping and lerps.

**Fix.** Bind the input buffer via `cudaTextureObject_t` (linear filtering,
clamp-to-edge addressing, normalized coords) so hardware does the bilinear fetch and the
texture cache captures 2D locality. The stabilization warp and keystone warp kernels
would benefit from the same change.
