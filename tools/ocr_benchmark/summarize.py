"""Generate the development-set table without discarding unsuccessful variants."""

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows = []
for family in ("tesseract", "neural"):
    rows.extend(json.loads((ROOT / f"results/ocr_benchmark/{family}.json").read_text()))
groups = collections.defaultdict(list)
for row in rows:
    groups[row["model"]].append(row)
lines = [
    "| Model | Baseline correct / 8 | Best explored / 8 | Blank correct? | Best setting |",
    "|---|---:|---:|---|---|",
]
for model, variants in groups.items():
    best = max(variants, key=lambda r: (r["correct"], -r["blank_false_positives"]))
    baseline = next(
        (
            r
            for r in variants
            if r["setting"]
            in ("oem1/psm8/otsu/scale1", "rgb", "rgb/invertFalse", "otsu/invertTrue")
        ),
        variants[0],
    )
    blank = next(r for r in best["readings"] if r["expected"] == "")
    lines.append(
        f"| {model} | {baseline['correct']} | {best['correct']} | {'yes' if blank['correct'] else 'no'} | `{best['setting']}` |"
    )
(ROOT / "results/ocr_benchmark/table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
