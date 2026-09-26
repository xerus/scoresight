# Feature Requests

This list tracks requests raised while testing the Windows portable build. Keep each item scoped to one visible behavior and add a regression test where the image-processing result can be checked without a camera.

## H.Scale for OCR targets

**Status:** Implemented on the feature branch; Windows packaging pending.

Add an **H.Scale** control beside **V.Scale** in Target Information Settings. It saves a value per detection box, leaves the OCR image unchanged at 10, and widens tightly spaced characters before OCR at higher values (up to 20). Keep existing V.Scale behavior and saved settings from older configurations intact. Verify the OCR input size, setting persistence, and UI wiring. Scaling cannot separate pixels that are already joined in the source image.

## Force Format

**Status:** Behavior needs a decision.

The disabled **Force Format** checkbox is currently a placeholder with no handler. The adjacent **Format** regex already rejects OCR text that does not fully match, so simply enabling the checkbox would duplicate existing behavior. An isolated change to remove the placeholder and explain the Format filter is ready for review. If the intended feature is automatic correction, define it with concrete before/after examples (for instance, which characters may be changed and how ambiguous matches are handled). Do not silently alter OCR values without that rule.

## Delivered in the current feature branch

- **Results (No Names)** keeps box outlines and OCR values visible while hiding box labels, including during selection.
- **Czech interface translation** adds Čeština to the language menu and bundles its compiled catalog in Windows builds.
