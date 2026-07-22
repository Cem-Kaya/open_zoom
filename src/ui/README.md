# ui module

This module owns the Qt widget layer.

Current contents:
- `MainWindow` for the persistent render surface, three auto-fading Simple-mode
  corner clusters, numbered mode grid/toast, and right-side Advanced Image and
  Assistant tabs with section arrows, labeled top-level AI Settings, diagnostics, and
  Chat/History workflows
- `AiSettingsDialog` for Codex subscription or OpenAI-compatible provider,
  model/reasoning defaults, shared language/behavior instructions, Advanced
  Assistant permissions, scene prompting, OCR, installed Windows voice/speed
  selection, manual speech preview, and notes configuration
- `RenderWidget` for native D3D12 presentation
- `AssistiveOverlay`, an owned floating Assistant that uses native move/resize
  handling without letting streamed text reapply its geometry, plus
  screen-reader streaming text, a follow-up question field, manual Read Aloud,
  high-contrast Close, and Escape dismissal
- `JoystickOverlay` for on-canvas panning input

The UI layout and settings-ownership contract is recorded in
[`docs/ui_modes_design.md`](../../docs/ui_modes_design.md).

Input routing from the window into app-level behavior stays split between this module and `src/app/interaction_controller.cpp`.
