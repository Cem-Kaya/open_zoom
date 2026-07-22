# Simple and Advanced UI Design

This note records the intended main-window structure and settings ownership.
It is the reference for future UI work so the live magnified image does not
gradually lose space as controls are added.

## Primary Goal

The camera image is the product's primary surface. Controls must remain easy
to find and large enough for a low-vision user without pushing the image below
the fold or surrounding it with permanent tool panels.

## Simple Mode

Simple mode is the normal operating view:

- The live camera fills the complete client area. Three small control clusters
  sit flush to its corners: mode switch at top-left, profile navigation at
  bottom-left, and immediate actions at bottom-right. Processing status is
  intentionally absent from Simple mode and lives in Advanced diagnostics.
- The clusters are frameless windows owned by the main window. This keeps them
  above the native D3D swap chain while retaining normal Qt buttons,
  accessibility metadata, focus, and tooltips.
- Each cluster uses an opaque background and a 3 px high-contrast border so it
  remains legible over every possible camera image.
- Chrome fades after five seconds without input. Mouse movement, a mouse
  button, wheel input, a key press, focus entry, or application activation
  reveals it immediately. Focused controls and an open mode grid remain
  visible. `Ctrl+H` pins the controls on screen or restores automatic hiding.
- The current profile is flanked by previous/next buttons. The grid button or
  current-profile button opens a temporary tile grid with plain-language names
  and number badges.
- Number keys `1` through `7` apply the first seven quick modes from anywhere
  in Simple mode. `Tab` and `Shift+Tab` move across the separate corner
  clusters in a predictable order, and `Esc` closes the mode grid.
- View navigation (wheel, keyboard, joystick, and middle-drag pan/zoom) updates
  the live focus without reclassifying the active mode as a custom setup. A
  real processing-control edit in Advanced still creates a custom setup.
- A profile change displays a large centered toast and sends an assertive Qt
  accessibility announcement. It does not start text-to-speech. The toast
  disappears automatically.
- Built-in modes use action-oriented names such as "Read a Page", "Keep It
  Steady", and "See in Low Light"; internal preset names remain unchanged for
  persistence and configuration lookup.
- The immediate actions are Photo, Record, Explain, and Read.
- Device selection and detailed numeric controls are intentionally absent.

Adding a new profile must not create persistent chrome. It joins the mode grid
and remains reachable by scrolling; only the first seven entries receive
number shortcuts.

## Advanced Mode

Advanced mode keeps the live image visible and opens a narrow inspector on its
right. Its top-level tabs are `Image` and `Assistant`; previous/next arrows
wrap across current and future sections, while a top-level pop-out button opens
AI Settings without placing service configuration inside the Image form.

The scrollable Image inspector has two ownership groups:

1. **Global device** contains the camera, physical orientation, and discovered
   camera formats.
2. **Current profile** contains all image-processing and assistive-mode values,
   plus the command that saves the current configuration as a quick option.

The inspector is constrained to 380-520 pixels. Detailed controls scroll
inside it instead of increasing the height of the main control area.

Assistant is a separate work surface rather than another image-processing
section. It shows Codex/ChatGPT connection and usage state, a camera-aware chat,
and a history tab. A new conversation can attach the current processed frame;
follow-up questions can keep or omit that attachment. Simple Explain creates a
temporary thread and never appears in history. Advanced Assistant creates
persistent threads and lists only ids created by OpenZoom, with resume, rename,
export, and delete actions. Codex owns the transcript store while OpenZoom keeps
the small title/preview/timestamp index in `settings.json`. Internet and coding
are explicit global Assistant permissions in AI Settings. Coding requires a
workspace folder and affects only persistent Advanced Assistant turns; Simple
Explain remains restricted even when those permissions are enabled.

Simple Explain and OCR results use one solid floating Assistant over the
camera. The panel preserves incremental streaming, exposes its text through a
focusable read-only text view, and provides a high-contrast Close control and
manual Read Aloud action. Its header is a drag handle, its edges and corners
resize within the camera bounds, and its question field attaches the current
view to the shared persistent Assistant conversation. Closing it (or pressing
Escape while it has focus) keeps the current result hidden until the next
user-requested analysis. OCR, scene explanations, and mode changes never start
speech automatically; only Read Aloud or the AI Settings Preview action does
so.

## Settings Ownership

Global values describe the device, application, or external services and do
not change when a quick profile is selected:

| Global value | Reason |
| --- | --- |
| Camera selection | A physical capture source is an application choice. |
| Camera orientation | Describes how the physical camera is mounted. |
| Simple/Advanced state | Restores the user's preferred working view. |
| Inspector collapsed state | UI preference, not image treatment. |
| Virtual joystick visibility | Interaction preference. |
| Selected quick profile | Restores the active workflow. |
| VLM/OCR endpoints, credentials, language, Read Aloud voice/speed, and note options | Service configuration is shared by profiles. |

Profile values describe how the current image should be treated and are saved
when the user creates a quick option:

| Profile value | Examples |
| --- | --- |
| Magnification | Zoom enabled, amount, and focus position. |
| Image cleanup | Black and white threshold, blur, and temporal smoothing. |
| Jitter reduction | Stabilization enabled and strength. |
| Display treatment | Color mode, contrast, and brightness. |
| Sharpening | Backend, enabled state, and strength. |
| Assistive behavior | OCR, scene explanation, and assistive overlay enabled states. |
| Diagnostics | Debug view and focus marker. |

`AdvancedConfig::rotationQuarterTurns` remains readable from old profile JSON
only for backward migration and is no longer written into profiles. Current
code persists and applies orientation at the top level of
`PersistentSettings`; profile comparisons ignore the legacy field.

## Layout Invariants

- Switching modes must not recreate or hide the render surface.
- Simple mode leaves the render surface full-size and overlays only the three
  corner clusters described above.
- Advanced controls live beside the camera, never above it as a tall form.
- Processing status belongs under Advanced Image diagnostics. Its visible
  value is deliberately short (`GPU Ready`, `Camera Offline`, `CPU Debug`, or
  `GPU Required`); full processing/error detail remains in its tooltip.
- Every interactive control keeps an accessible name, description, visible
  keyboard focus, and a logical tab order.
- New global controls belong in the Global device section. New processing
  controls belong in Current profile and must participate in config
  persistence and equivalence checks.
