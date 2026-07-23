- [x] Touchpad two-finger pan now updates the zoom center while zoom is active (handled via precision wheel gestures without modifiers).
- [x] Added a "Show Focus Point" debug checkbox that draws a red marker at the live zoom center, keeping it in sync with cursor-driven updates.
- [x] Switching cameras can trigger a crash (likely during CUDA surface reinitialization). Fixed: the CUDA stream is now synchronized before surface/buffer teardown, and the presenter drains the graphics queue before the shared texture is released. See `improvement_ideas/01-stability-threading.md` (S1).
- [x] Expand GPU logging even further while porting remaining pipeline stages to CUDA. Done: `cuda_interop.cpp` now logs cudaExternalMemory import parameters (size/flags), and color conversion + rotation moved onto CUDA. The optional break-on-fail hooks were never added and are dropped from this item.
- [x] Prototype a lightweight vision-language model overlay that can describe
      magnified regions and provide contextual hints for visually impaired
      users. Shipped as the assistive runtime: local Tesseract OCR, Codex /
      OpenAI-compatible VLM scene explanations, a streaming floating Assistant
      panel, Read Aloud, and lecture notes (see `src/common/assistive_runtime.cpp`).
