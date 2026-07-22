# Performance: GPU & CPU Pipeline

For a real-time magnifier, latency *is* the product. The single biggest win is P1.
Measure before/after each change — add the frame-timing instrumentation (P8) first so
improvements are quantified, not assumed.

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

- **Priority:** MEDIUM · **Effort:** large · **Status:** Reported
- **Evidence:** `src/cuda/cuda_kernels.cu` — `GaussianBlurHorizontalKernel` / `GaussianBlurVerticalKernel` (~334–402) read neighborhoods straight from global memory; radius can be up to 50.

**Problem.** Each output pixel reads up to 101 global-memory samples with no staging;
adjacent threads re-read overlapping windows. At 1080p with a large radius this is the
most expensive kernel in the pipeline by far.

**Fix.** Cooperative tile loading into `__shared__` memory (block width + 2×radius),
`__syncthreads()`, then filter from shared memory. Alternative with better
effort/benefit ratio for very large radii: replace with 2–3 iterated box blurs
(O(1) per pixel via running sum), which converges to Gaussian and makes radius cost-free.
Benchmark both against the current kernel before committing.

---

## P5. Cache Gaussian kernel weights; avoid re-upload on slider moves

- **Priority:** MEDIUM · **Effort:** small–medium · **Status:** Reported
- **Evidence:** `src/cuda/cuda_kernels.cu` — `UploadGaussianKernel()` (~691) recomputes weights and calls synchronous `cudaMemcpyToSymbol` whenever radius/sigma changes.

**Fix.** Cache the last-uploaded (radius, sigma) pair and skip the upload when unchanged
(likely a 5-line fix); optionally precompute the discrete radii set (0–50 per README)
at init. Verify first whether the current code already skips unchanged pairs.

---

## P6. Device-buffer reallocation churn on resolution change

- **Priority:** LOW–MEDIUM · **Effort:** medium · **Status:** Reported
- **Evidence:** `src/cuda/cuda_interop.cpp` — `EnsureDeviceBuffers()` (~302) frees and reallocates buffers A/B/scratch whenever dimensions change.

**Problem.** Resolution changes (mode switch, camera switch) free + `cudaMallocPitch`
three buffers mid-pipeline. Not per-frame today (caching exists ~312–316), so this is
lower priority than the agents' original report suggested.

**Fix.** Allocate at max-supported resolution once per surface lifetime, or use a CUDA
memory pool (`cudaMallocAsync`, CUDA 11.2+). Only worth doing after P1–P3.

---

## P7. CPU pipeline: per-frame allocations and full-frame copies

- **Priority:** MEDIUM · **Effort:** medium · **Status:** Reported
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

- **Priority:** MEDIUM · **Effort:** medium · **Status:** Reported (no timing exists today)
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
