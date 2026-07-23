# Third-Party Licenses and Attributions

OpenZoom relies on several third-party SDKs and code drops. Keep this summary with any redistributed build and update it whenever dependencies or bundled notices change.

## Qt 6
- Upstream: <https://www.qt.io/>
- License family: LGPL-3.0 / GPL-3.0 / commercial, depending on how Qt is obtained
- OpenZoom usage: dynamically linked Qt Widgets runtime deployed via `windeployqt`
- Notes:
  - If you redistribute Qt DLLs, include the required Qt notices and license text that apply to your distribution.
  - If you modify Qt itself, those changes must be handled under Qt's licensing terms.
  - Development builds expect the user to provide a local Qt installation.

## NVIDIA Image Scaling
- Upstream: <https://github.com/NVIDIAGameWorks/NVIDIAImageScaling>
- Local notice file: [`third_party/nvidia_nis/LICENSE.txt`](../third_party/nvidia_nis/LICENSE.txt)
- Usage: optional CUDA spatial sharpening path
- Attribution: retain NVIDIA's notice when redistributing modified or bundled source material

## AMD FidelityFX Super Resolution 1.0
- Upstream: <https://github.com/GPUOpen-Effects/FidelityFX-SuperResolution>
- Local notice file: [`third_party/amd_fsr1/LICENSE.txt`](../third_party/amd_fsr1/LICENSE.txt)
- License: MIT
- Usage: optional FSR-style CUDA spatial sharpening path
- Attribution: preserve the MIT license text and AMD attribution in redistributed source packages

## NVIDIA CUDA Toolkit
- Upstream: <https://developer.nvidia.com/cuda-toolkit>
- License: NVIDIA CUDA Toolkit EULA
- OpenZoom redistribution scope:
  - the application links the CUDA runtime statically
  - no standalone CUDA Toolkit runtime DLLs, headers, compilers, or developer
    tools are copied into release bundles
- Review NVIDIA's current redistribution terms before shipping CUDA-enabled
  binaries.

## NVIDIA Maxine Video Effects SuperRes
- Upstream headers: <https://github.com/NVIDIA/MAXINE-VFX-SDK>
- Runtime download page:
  <https://www.nvidia.com/en-me/geforce/broadcasting/broadcast-sdk/resources/>
- Local notices: [`third_party/maxine/Maxine-VFX-SDK/LICENSE`](../third_party/maxine/Maxine-VFX-SDK/LICENSE)
  and [`third_party/maxine/LICENSE.txt`](../third_party/maxine/LICENSE.txt)
- Header/sample snapshot license: MIT
- Runtime license: NVIDIA SDK License Agreement; the supported 0.7.6 Video
  Effects runtime is obtained and installed separately by the user.
- OpenZoom usage: optional runtime-loaded SuperRes on supported NVIDIA GPUs.
  The GPL application resolves `NVVideoEffects.dll` and `NVCVImage.dll` with
  `LoadLibrary`/`GetProcAddress`; it has no import-library dependency.
- Distribution scope: no NVIDIA Video Effects runtime binaries, models, or
  installers are stored in the repository or copied into OpenZoom bundles.
  The Setup Assistant fetches a pinned installer directly from NVIDIA, verifies
  its SHA-256 value, and launches NVIDIA's installer after the user chooses to
  install it.
- Required product attribution: `SuperRes powered by NVIDIA Maxine™`.
  This attribution does not imply NVIDIA endorsement.

## Tesseract OCR Windows Runtime
- Upstream engine: <https://github.com/tesseract-ocr/tesseract>
- Windows distribution: <https://github.com/UB-Mannheim/tesseract/wiki>
- Tesseract license: Apache-2.0
- OpenZoom usage: optional local OCR process. The Setup Assistant downloads a
  pinned UB Mannheim installer directly from the distributor, verifies its
  SHA-256 value, and installs a user-managed copy under
  `%LOCALAPPDATA%\OpenZoom\tools\tesseract` only after the user requests it.
- Distribution scope: OpenZoom release bundles contain no Tesseract binary,
  language data, installer, or transitive runtime library.
- The Windows runtime dynamically links Leptonica plus image, compression,
  Unicode, font, and GNU runtime libraries shipped by the UB Mannheim build.
  Those components retain their own permissive, LGPL, or runtime-exception
  terms. The user accepts the distributor and dependency terms when choosing
  to install it; the Setup Assistant can remove only OpenZoom's managed copy.

## Lucide Icons
- Upstream: <https://lucide.dev/>
- Package/version: `lucide-static` 1.25.0
- Local notice file: [`assets/icons/lucide/LICENSE`](../assets/icons/lucide/LICENSE)
- License: ISC; several inherited Feather icons also carry the MIT notice in
  the same license file
- OpenZoom usage: embedded Qt resource icons for camera actions, floating
  Assistant controls, Advanced section navigation, and keystone history
  Previous/Stop/Continue/Next actions
- Attribution: retain the local license file in source and the third-party
  notice in redistributed builds

## Microsoft Platform Components
- APIs used: Media Foundation, Direct3D 12, DXGI, Windows Imaging Component, and other Windows SDK libraries
- License source: Windows SDK / Visual Studio / OS redistribution terms
- Notes: these platform APIs are not copied into the repository as standalone third-party source code

## Not Currently Bundled
- OpenCV DNN
- TensorRT
- external AI model weights
- NVIDIA Video Effects runtime, models, and installers
- Tesseract OCR runtime, language data, and installer
- OpenAI Codex CLI binaries and installer/bootstrap files

`OPENZOOM_ENABLE_TEXT_SR` adds only the dynamic Maxine adapter built from the
MIT header snapshot. It does not add a link-time or redistribution dependency
on the proprietary runtime.

## Optional External Tools And Services
- OpenAI-compatible VLM services may be used at runtime through user-supplied endpoint credentials, but no hosted model or service SDK is bundled here.
- OpenAI Codex CLI may be launched as an optional external `codex app-server`
  process. At the user's request, Setup Assistant may download an exact pinned
  copy of OpenAI's official Windows bootstrap script, verify its SHA-256, and
  run it with prompts disabled. That upstream bootstrap independently verifies
  the selected official release package against OpenAI's checksum manifest.
  OpenZoom does not bundle Codex source, binaries, model weights, installer, or
  an OpenAI SDK. Codex authentication, service access, usage limits, updates,
  and licensing remain governed by the user's Codex installation and OpenAI
  terms.
