# Build System, Tooling & Docs Hygiene

Guardrails that make every other change in this backlog safer, plus documentation
cleanup. Repo hygiene note: `build/` exists on disk but is **not** tracked in git
(verified — 58 tracked files, none under `build/`), so no action needed there.

---

## B1. Enable compiler warnings

- **Priority:** MEDIUM · **Effort:** small · **Status:** Confirmed (no `target_compile_options` warning flags anywhere in cmake/)
- **Evidence:** `cmake/ProjectOptions.cmake`, `cmake/CMakeLists.txt`

**Fix.** Add `/W4` (MSVC) to the `open_zoom` target; fix the fallout (expect mostly
unused-variable and conversion warnings); then add `/WX` so it stays clean. Scope flags
to project targets only — don't let them leak into `third_party/` (amd_fsr1, nvidia_nis).
For the `.cu` files, pass host-compiler warnings via
`$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>` or exclude them initially. Do this *before*
the app.cpp decomposition so the refactor is warning-guarded.

---

## B2. Create the missing test infrastructure

- **Priority:** MEDIUM · **Effort:** large (but start small) · **Status:** Confirmed (an `OPENZOOM_ENABLE_TESTS` option exists but `tests/` does not; cmake warns about it at cmake/CMakeLists.txt ~102–108)
- **Evidence:** cmake option present, zero test files in the repo

**Problem.** 8,800 lines of pipeline/persistence logic with no tests; every refactor in
this backlog is currently verified only by running the app on a Windows box with a
camera.

**Fix.** Don't aim for coverage — aim for the pure, hardware-free logic first:
1. `tests/CMakeLists.txt` with Catch2 (FetchContent) wired to the existing
   `OPENZOOM_ENABLE_TESTS` option and `ctest`.
2. First targets, in order of value: `image_processing` (NV12/YUY2 conversion with known
   pixel fixtures — also locks in the V2 validation work), `settings_store`
   (round-trip, legacy v1, corrupt file, out-of-range fields), zoom/crop rect math.
3. Add a `msvc-test` preset and make `run_minimal_test.bat` (currently references a
   possibly-unwired `dx12_cuda_minimal` target — verify) run ctest with a clear
   pass/fail exit code.

Capture/D3D12/CUDA stay manual-test-only for now; document that in `tests/README.md`.

---

## B3. Minimal CI pipeline

- **Priority:** MEDIUM · **Effort:** medium · **Status:** Confirmed (no `.github/` directory exists)

**Fix.** `.github/workflows/build.yml` on `windows-latest`: install Qt via
`jurplel/install-qt-action`, configure with `-DOPENZOOM_ENABLE_CUDA=OFF` (CUDA toolchain
is impractical on hosted runners), build the `msvc-cpu` preset, run ctest (after B2).
A CPU-only compile+test gate already catches the majority of breakage in this repo —
CHANGELOG shows CPU-only builds have broken before ("CPU-only CMake definitions now
honor OPENZOOM_ENABLE_CUDA=OFF").

---

## B4. clang-format (and later clang-tidy)

- **Priority:** LOW–MEDIUM · **Effort:** small · **Status:** Confirmed (no config files exist)

**Fix.** Add a `.clang-format` matching the *existing* style (4-space indent, attached
braces, ~120 columns — derive from current sources rather than imposing LLVM defaults,
to keep diffs small); optionally one whole-repo format commit. Add `.clang-tidy` later
with a narrow check set (`bugprone-*`, `performance-*`, `concurrency-*`) — the analyses
in `01`/`05` are exactly what those checks automate.

---

## B5. Deduplicate hardcoded Qt/CUDA paths

- **Priority:** MEDIUM · **Effort:** medium · **Status:** Reported
- **Evidence:** `C:\Qt\6.9.3\msvc2022_64` hardcoded in `cmake/CMakePresets.json` (3 presets) and both build scripts; CUDA v13.0 path in `scripts/build_release_bundle.bat` (~20–21); `docs/hardcoded_paths.md` documents these

**Fix.** Single source of truth: presets read `$env{QT_PREFIX}` with the current path as
fallback; scripts set/respect `QT_PREFIX` and `CUDA_PATH` env vars instead of embedding
paths. Add a `scripts/validate_environment.bat` that checks `cl.exe`, Qt, CMake ≥3.23,
and (optionally) CUDA before configuring, and print actionable remediation messages.
Update `docs/hardcoded_paths.md` to describe the override mechanism instead of the
literal paths.

---

## B6. Docs cleanup and consolidation

- **Priority:** LOW · **Effort:** small · **Status:** Confirmed (files listed exist)

Items:
1. `docs/chatgpt_future_readme.txt` — legacy scratch notes with outdated status claims
   mixed in. Extract still-valid TODOs into `docs/progress.md`, then delete it (git
   history preserves it). Its presence actively misleads doc-first agents (per
   `agents.md` workflow).
2. `docs/rotation_ui_notes.md` — unreferenced from any documentation map; either link it
   from `docs/README.md` or fold its content into `docs/code_reference.md`.
3. `TODO.md` (root) vs `docs/progress.md` vs `docs/ai_upscaling_todo.md` — three
   overlapping trackers. Pick one canonical tracker (suggest `TODO.md` for active items,
   `docs/progress.md` for completed-milestone history), and say so in `agents.md`.
4. Root `CMakePresets.json` is a 4-line include shim for `cmake/CMakePresets.json` — add
   a `"$comment"` field noting "edit cmake/CMakePresets.json, not this file". Note:
   README.md line 54 says presets live in `cmake/CMakePresets.json` — keep that accurate.
5. `agents.md` module map says `src/d3d12/`, `src/capture/`, `src/ui/`, `src/common/`
   are "reserved for the ongoing refactor; consult their README stubs" — they are fully
   populated and no README stubs exist. Update the map to match reality (this file is
   the first thing agents read; staleness there is costly).

---

## B7. Batch-script robustness

- **Priority:** LOW · **Effort:** small · **Status:** Reported
- **Evidence:** `scripts/build_and_run.bat`, `scripts/build_release_bundle.bat` — no toolchain validation before configure; generic failure messages

**Fix.** Covered mostly by B5's `validate_environment.bat`. Additionally: `exit /b`
with distinct codes per failure stage, and echo the chosen Qt/CUDA paths at start so
mis-resolution is visible in the first lines of output.
