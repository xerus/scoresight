# Reading bright scores on an illuminated panel

The October 31 recording at 17:48 shows home score **59** and away score **17**.
Glare joins the away digits into a wide silhouette. The neural reader's native
and shear variants frequently disagree, and changing the box margins changes
the result. The old reader sometimes accepts only `1`.

For numeric fields with an unknown or single-digit neural reading and evidence
of multiple joined digits, PP-OCR now tries three brighter core masks. Each mask
is read with the original margins and with normalized margins around its bright
content. Two different thresholds must confidently agree on a complete number;
two margin variants of one threshold are not enough. Unknown frames remain
unknown. This retry works with the optional LED check both enabled and disabled.

The user's recording supplied 21 lossless regression crops, with permission.
Tests cover seven positions/sizes of the away box, including a wider box and
five-pixel shifts. At frames 32 and 164 all seven boxes must acquire `17` with
the LED check either on or off. Other selected frames may be rejected but must
never return a shortened or different score. The home-score crops retain `59`.

An additional diagnostic sampled every tenth frame through the 328-frame clip,
using seven boxes (231 crops total), with the optional LED check disabled:

| Result | Before | After |
| --- | ---: | ---: |
| Correct `17` | 12 | 127 |
| Incorrect `1` | 9 | 0 |
| Unknown | 210 | 104 |

All seven boxes acquire the score by frame 10 (about 0.33 seconds) in this
sample. Previously, the wide box and upward-shifted box never acquired it.
The bounded retry adds CPU work: median recognition times in this run increased
from roughly 34–43 ms to 81–89 ms per crop. These are development measurements
on this recording, not a general accuracy or performance guarantee. Raw counts,
box coordinates, frame indices, timing, and the baseline commit are recorded in
[panel_scoreboard.json](../results/ocr_video_benchmark/panel_scoreboard.json).

JSON consumers now retain a field's last accepted nonempty value through empty
and rejected readings, so remaining unknown frames do not clear a score already
acquired. Full JSON still exposes the current detection state. See the
[HTTP JSON contract](http_server.md) and [outbound API guide](out_api.md).
