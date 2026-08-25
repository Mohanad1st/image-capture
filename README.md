# image-capture

A small FastAPI service that grabs a single frame from an RTSP camera, detects water meters in it
with a YOLO model, and returns **cropped images of each meter** for a person to read.

It does not read the meter. That is the point.

## Why it returns images, not numbers

The obvious version of this service would return `{"meter_1": 4718.2}`. This one returns a cropped
JPEG instead, and a human enters the value.

That is a deliberate design boundary, not an unfinished feature:

- **A wrong number is invisible; a wrong crop is obvious.** If OCR misreads a digit, the error enters
  the record silently and downstream consumption maths inherits it. If the crop is blurry, occluded, or
  framed on the wrong dial, the person looking at it can see that immediately and reject it.
- **Accountability stays with a named person.** The reading is attributable to whoever entered it, not
  to a model version that may since have been replaced.
- **The failure mode is refusal, not fabrication.** No detections returns `404` rather than a guess.
  The service is allowed to say "I could not see a meter."

The model's job is narrowed to the thing it is actually reliable at — *finding* the meter in a wide
camera frame. Interpretation, which carries the consequence, stays with the operator.

## API

Single endpoint:

```
GET /snapshot?rtsp_url=rtsp://user:pass@host:554/stream
```

**200** — one entry per detected meter, JPEG bytes base64-encoded:

```json
{ "meter_1": "<base64 jpeg>", "meter_2": "<base64 jpeg>" }
```

| Status | Meaning |
|---|---|
| `200` | one or more meters detected and cropped |
| `404` | frame captured, no meters detected — not an error, an honest negative |
| `500` | model weights not loaded on the server |
| `502` | RTSP stream could not be opened, or the frame could not be read |

The RTSP URL is a request parameter and is never stored or logged by this service. It usually contains
camera credentials, so treat the caller as trusted and terminate TLS in front of it.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                                        # serves on :8585
```

Verify OpenCV resolved to the headless build — see the note at the top of `requirements.txt`:

```bash
python -c "import cv2; print(cv2.__file__)"
```

### Model weights

`best.pt` is **not in this repository** and is not distributed. It is a YOLO detector trained
separately on meter imagery from the specific installations this runs against; publishing it would
imply a generality it does not have.

Place your own `best.pt` next to `run.py`. Without it the service starts and answers `500` on
`/snapshot` rather than crashing — degraded, but honest about being degraded.

## Deployment

Pushes to `main` deploy to Azure Web App `camera-capure` via
`.github/workflows/main_camera-capure.yml`, authenticating with OIDC federated credentials (no stored
publish profile). The weights are uploaded to the App Service filesystem alongside `run.py`, separately
from the repository.

Two things to know before you touch it:

- **`ultralytics` pulls in `torch`.** The install is large. Confirm the App Service plan has the disk
  and memory headroom before assuming a failed deploy is a code problem.
- **The Azure app name contains a typo** (`camera-capure`, missing the `t`). It is left as-is on
  purpose: the name is bound to the live App Service and the OIDC federated credential subject, so
  renaming breaks deployment for a cosmetic gain.

## What this is not

Not a metering system of record, not a billing input, and not a substitute for a calibrated meter
reading. It is a capture-and-crop service that shortens the walk to the meter while leaving the
reading — and responsibility for it — with a person.

## License

MIT — see [LICENSE](LICENSE).
