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
  - runtime DLLs only
  - no CUDA headers, compilers, or developer tools are bundled
- Review NVIDIA's current redistribution terms before shipping binaries that include CUDA runtime files

## Tesseract OCR Windows Runtime
- Upstream engine: <https://github.com/tesseract-ocr/tesseract>
- Windows distribution: <https://github.com/UB-Mannheim/tesseract/wiki>
- Tesseract license: Apache-2.0
- OpenZoom usage: optional local OCR process; release bundles copy the installed
  runtime, English/orientation language data, and the distribution's Tesseract
  license/authors/readme under `tools/tesseract/`
- The Windows runtime dynamically links Leptonica plus image, compression,
  Unicode, font, and GNU runtime libraries shipped by the UB Mannheim build.
  Those components retain their own permissive, LGPL, or runtime-exception
  terms. Keep all DLLs replaceable and complete a transitive-license notice
  audit before commercial redistribution outside this development bundle.

## Lucide Icons
- Upstream: <https://lucide.dev/>
- Package/version: `lucide-static` 1.25.0
- Local notice file: [`assets/icons/lucide/LICENSE`](../assets/icons/lucide/LICENSE)
- License: ISC; several inherited Feather icons also carry the MIT notice in
  the same license file
- OpenZoom usage: embedded Qt resource icons for camera actions, floating
  Assistant controls, and Advanced section navigation
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

Those remain roadmap items. If they are added later, this file must be expanded in the same change.

## Optional External Tools And Services
- OpenAI-compatible VLM services may be used at runtime through user-supplied endpoint credentials, but no hosted model or service SDK is bundled here.
- OpenAI Codex CLI may be launched as an optional external `codex app-server`
  process. OpenZoom does not bundle Codex source, binaries, model weights, or an
  OpenAI SDK. Codex authentication, service access, usage limits, updates, and
  licensing remain governed by the user's Codex installation and OpenAI terms.
