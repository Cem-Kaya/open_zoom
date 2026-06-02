# ui module

This module owns the Qt widget layer.

Current contents:
- `MainWindow` for the control panel and render surface layout
- `RenderWidget` for native D3D12 presentation
- `JoystickOverlay` for on-canvas panning input

Input routing from the window into app-level behavior stays split between this module and `src/app/interaction_controller.cpp`.
