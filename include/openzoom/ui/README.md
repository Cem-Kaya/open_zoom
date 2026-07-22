# ui public headers

Public Qt UI interfaces live here.

Currently exported:
- `main_window.hpp`: the persistent render surface plus the auto-fading
  Simple-mode corner controls, mode announcements, and right-side Advanced
  Image/Assistant inspector interfaces
- `ai_settings_dialog.hpp`: provider, Codex model/reasoning and
  permissions/workspace, shared Assistant Instructions, scene VLM prompt, OCR,
  installed Windows voice/speed, manual speech preview, and notes settings
- `main_window.hpp`: also declares the native-movable/resizable floating
  Assistant and its question, Read Aloud, Close, and focus-loop interfaces

See [`docs/ui_modes_design.md`](../../../docs/ui_modes_design.md) for the layout
and global-versus-profile settings contract.
