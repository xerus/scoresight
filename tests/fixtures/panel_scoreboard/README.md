# Illuminated panel scoreboard

Lossless crops from the user's own recording, included with their permission:
`Recording from 2026-10-31 17-48-28.861960.mp4` (30 fps, 1920×1080).

The visually labeled scores are home **59**, away **17** throughout these samples.
`manifest.json` records source frame indices and original pixel coordinates.
The `context_*.png` crops use source ROI `[1570, 240, 100, 100]` and allow tests
with tighter, wider, and shifted detection boxes. No resizing or enhancement
was applied to the fixtures. The bright digits bleed together on the green
panel; changing box margins previously changed neural readings substantially.

The `recover` cases require acquisition. Other samples may be unknown but must
never publish a shortened or different score. These are development regressions,
not an independent estimate of OCR accuracy.
