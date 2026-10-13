# Basketball field settings

Select a field, then choose **Box reader** to inherit the global OCR Model or
use a bundled reader for that field. This allows LED scores and neural/Tesseract
clocks to coexist. The external Tesseract model picker remains global; select
**Use global model** for fields using it.

Readers load on first use and stay cached. A failed optional reader rejects only
its own fields and retries after five seconds; other readers keep producing
results. The application log contains the load error. Changing a box's reader
or settings starts confirmation again.

The **Basketball** button applies a format and confirmation policy to the selected
standard field:

| Field | Accepted format | Confirmation |
| --- | --- | --- |
| Home / Away Score | One to three digits | Numbers / scores |
| Time | M:SS, MM:SS, or seconds with one decimal | Countdown |
| Shot Clock | Integer seconds up to 39 or seconds with one decimal | Countdown |
| Period | One or two digits, including numeric overtime periods | Numbers / scores |
| Home / Away Fouls | One or two digits | Numbers / scores |

The button preserves ROI coordinates, reader choice, LED calibration, confidence,
ordinal display, and output mapping. It disables Average Output and Skip Similar
Image, and enables Skip Empty Values. It applies to standard field names and is
disabled for custom names. Position the boxes and choose readers for the actual
display separately. The preset does not infer digit count or restrict basketball
to a particular league's period duration or shot-clock reset rules.

PP-OCR + TTA and Tesseract can read clock text. The OpenCV reader supports clocks
with [explicitly calibrated digits and visible separators](calibrated-led-clocks.md).
Its default Number layout rejects Time fields.
See [confirmation behavior](temporal-confirmation.md) for update delay and resets,
and [LED calibration](opencv-benchmark.md) for digit geometry and color settings.
