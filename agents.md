# Agent Operation Guide

OpenZoom evolves as an **AI-assisted, GPU-accelerated magnifier** for people
with low vision. Agents help us deliver that mission by building features,
keeping the pipeline fast, and ensuring the docs stay accurate. Read this
guide end-to-end before you touch the repo.

This project expects any autonomous or semi-autonomous agent (LLM, scripting
bot, CI assistant) to work within the following guardrails. Keep this document
open while contributing and update it whenever the workflow evolves.

## Core Principles
1. **Read the docs first** – Always consult `README.md`, `docs/README.md`,
   `docs/hardcoded_paths.md`, and `docs/THIRD_PARTY_LICENSES.md` before making
   changes. They describe the architecture, build matrix, and licensing rules.
2. **Keep documentation current** – Any code or script change that affects
   usage, outputs, dependencies, or licensing must update the relevant doc.
   Never commit features or fixes without doc updates.
3. **Respect the dual license** – All contributions are accepted under GPL-3.0
   plus the commercial license. Do not add third-party code unless the license
   is compatible and you note it in `docs/THIRD_PARTY_LICENSES.md`.

## Module Map
- `src/app/` – Qt entry point and application wiring.
- `src/cuda/` – CUDA interop and kernels.
- `src/d3d12/`, `src/capture/`, `src/ui/`, `src/common/` – reserved for the
  ongoing refactor; consult their README stubs before populating them.
- Public headers mirror the source tree under `include/openzoom/`.

## Workflow Expectations
- Align every change with the mission: produce a responsive magnifier that
  helps visually impaired users read content with AI assistance (temporal
  smoothing, upcoming VLM overlays, adaptive sharpening).
- Maintain coding style and structure; mirror the source layout in
  `include/openzoom/…` and keep module READMEs updated as you populate them.
- Update `CHANGELOG.md` and licensing notices when shipping user-visible or
  legal-impacting changes.
- Ensure build scripts (`scripts/build_and_run.bat` and
  `scripts/build_release_bundle.bat`) remain functional on Windows 10/11 with
  the documented toolchain.
- Run the appropriate build or test command locally before submitting
  automated changes; note the result in your summary.

## Useful References
- Qt moc and object model: <https://doc.qt.io/qt-6/moc.html>
- NVIDIA CUDA interop (runtime API): <https://docs.nvidia.com/cuda/>
- Media Foundation capture overview: <https://learn.microsoft.com/windows/win32/medfound/>
- Direct3D 12 best practices: <https://learn.microsoft.com/windows/win32/direct3d12/>
- Accessible magnification guidance: <https://www.w3.org/WAI/standards-guidelines/>
- VLM-assisted magnification ideas: <https://arxiv.org/abs/2102.09576>

Agents that modify the workflow must append to this guide so future runs remain
aligned with the project goals.
