# NEXUS Smart Factory — official identity integration

## Authoritative visual assets

Source: connected Google Drive folder
`Logo/NEXUS_Identity_2_0` (folder ID `1rU3QCrdwCdHB0tYVLIq0W6aPqpj3iOsY`).

The approved design is the **third Figma variant: signature spectrum X**.
These two SVGs are unmodified exports of the owner's approved files:

| Repository asset | Source file | Use |
| --- | --- | --- |
| `static/nexus-official-wordmark.svg` | `NEXUS_Official_Transparent_Logo.svg` | Shared dashboard / case-study / settings sidebar |
| `static/nexus-official-x.svg` | `NEXUS_Official_X_Transparent.svg` | Favicon, Decision Studio rail and unobtrusive hero watermark |

The approved palette is midnight `#0B0F19`, deep blue `#061524`,
cyan `#00E0FF`, medium blue `#4EA8FF`, lavender `#8F9BFF`,
violet `#6E55C6`, and silver `#B9C5CE`.

No unofficial lettering or regenerated logo shape is substituted. The SVG
files are stored in-repository and served through Flask's own static path
so the app can render them without loading Drive, Figma or image CDNs.

## UI scope

- `layout.py` renders the official wordmark on every page that uses the
  shared sidebar, with a keyboard-focusable route back to the dashboard.
- `main.py`, `case_study.py`, `rul_case_study.py`, and `settings.py`
  set the same branded X favicon.
- `static/nexus-design.css` applies consistent spectrum accents, responsive
  logo sizing and focus-visible states. It preserves the stale-telemetry
  treatment and safety notices.
- `static/decision-studio.html` replaces its placeholder "N" block with
  the official X asset; `static/decision-studio.css` now uses matching
  midnight backgrounds, accessible light text and cyan/lavender highlights.

Branding is presentation only. **Risk predictions still use simulated AI4I
observations**, and the RUL estimate on the dashboard remains a heuristic
proxy rather than the separate real FD001 regression benchmark. The visual
changes do not connect to actual industrial equipment.

## Regression checks

```bash
python -m unittest discover -s tests -p test_nexus_brand_identity.py -v
```

Seven focused offline checks cover the SVG parser, palette, navigation
semantics, favicon consistency, Decision Studio parity, stale-state visibility
and Docker asset packaging. Registered in CircleCI `python-quality`.

Before merging: check that both logos actually render on desktop and mobile,
that focus states remain visible, validate contrast after Google Font fallback,
and require Python, browser and Docker CircleCI quality jobs on one exact
PR head. Render deployment commit and key endpoint responses must then
be verified independently.
