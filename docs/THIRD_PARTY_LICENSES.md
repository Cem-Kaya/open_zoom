# Third-Party Licenses and Attributions

OpenZoom depends on several external components. This document summarises
license requirements, attribution, and redistribution notes. Keep it with any
binary distribution (copied automatically by the release bundle script).

---

## Qt 6 (LGPL-3.0)
- Website: <https://www.qt.io/>
- License: GNU Lesser General Public License v3.0
- Requirements:
  - Use dynamic linking or provide object files to allow relinking.
  - Provide a copy of the LGPL-3.0 license and state that Qt is used.
  - Publish any modifications to Qt under the same license.
- Source: Qt binaries and headers are not bundled in this repository; users
  supply their own Qt installation.

## NVIDIA Image Scaling (NIS)
- Repository: <https://github.com/NVIDIAGameWorks/NVIDIAImageScaling>
- License: NVIDIA Source Code License for NIS (permissive)
- Attribution: “Contains NVIDIA Image Scaling technology. © NVIDIA Corporation.”
- Redistribution: Allowed in binary form; include the license text if you ship
  modified sources.

## AMD FidelityFX Super Resolution 1.0 (FSR1)
- Repository: <https://github.com/GPUOpen-Effects/FidelityFX-SuperResolution>
- License: MIT License
- Attribution: “FidelityFX Super Resolution by Advanced Micro Devices, Inc.”
- Redistribution: Maintain the MIT license header in source and documentation.

## NVIDIA CUDA Toolkit
- Website: <https://developer.nvidia.com/cuda-toolkit>
- License: CUDA Toolkit EULA (redistributables)
- Redistribution: Only the runtime DLLs permitted by the EULA are packaged.
  Include NVIDIA’s copyright notice and do not distribute development headers
  or tools outside the terms of the EULA.

## Other Dependencies
- Windows Media Foundation, Direct3D, and other Windows SDK libraries are
  provided by Microsoft and covered under the Windows SDK license. They are not
  redistributed with OpenZoom.

---

For the full license texts, see `LICENSE_GPL.txt`, `LICENSE_COMMERCIAL.txt`,
and the upstream license files linked above.
