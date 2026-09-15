# LB-anythings

A [Label Studio](https://labelstud.io) ML backend that proposes bounding boxes, learns from
the corrections, and retrains itself as the annotations pile up. It is detector-agnostic: the
same service serves any single- or multi-class box project. Point it at a checkpoint and a
data directory.

It replaces the `ls_backend` folder of the pipe-counting project, speaks the same protocol to
Label Studio, and reads that project's existing examples without conversion.

## What it does

- **Predicts.** Opening a task shows the detector's boxes, already carrying your project's
  label, for an annotator to correct rather than draw.
- **Collects.** Every annotation submitted or updated becomes a training example on disk.
- **Retrains.** Every 25 new examples (configurable), a training run starts on its own, and
  the next prediction uses the checkpoint it produced. No restart.
- **Trains on demand.** The Start Training button pulls every annotated task from the project
  and trains on all of them, including annotations made before the backend was connected.
- **Mines.** `lb-anythings mine` finds the frames of a video the detector is least sure about,
  so you label its blind spots instead of random frames.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU for training. Torch and torchvision are pinned to the CUDA 12.8 index in
  `pyproject.toml`, which is what a Blackwell card (RTX 50-series) needs. **Never reinstall
  torch from the default PyPI index**: you will get a CPU or old-CUDA build and lose the GPU.
  Use `uv add`.

## Install and run

```powershell
git clone https://github.com/pergazuz/LB-anythings
cd LB-anythings
uv sync                       # includes the ML stack: ultralytics, torch (cu128), opencv
uv run lb-anythings serve     # http://0.0.0.0:9090
```

The server starts with no checkpoint and serves empty predictions until one exists, so you can
connect it to a brand-new project and let the first training run create the detector.

To serve a checkpoint you already have, point at it:

```powershell
$env:LB_CHECKPOINT = "C:\path\to\best.pt"
uv run lb-anythings serve
```

## Connect it to Label Studio

In the project, **Settings → Model → Connect Model**:

- **Name:** anything, for example `lb-anythings`
- **Backend URL:** `http://localhost:9090`. If Label Studio runs in Docker, use
  `http://host.docker.internal:9090` instead: `localhost` inside the container is the
  container.
- **Authentication:** No Authentication
- **Interactive preannotations:** on, if you want predictions while labelling.

**Validate and Save** calls `GET /health` then `POST /setup`. Open a task and the boxes appear.

For the backend to fetch images you uploaded to Label Studio, it needs to reach Label Studio
back. Label Studio sends a hostname and access token with most requests; when it does not, set
them yourself:

```powershell
$env:LABEL_STUDIO_URL = "http://localhost:8080"
$env:LABEL_STUDIO_API_KEY = "<your access token>"   # Account & Settings -> Access Token
```

These are also what the **Start Training** button needs.

The labeling interface must have exactly one `RectangleLabels` control bound to exactly one
`Image`. Any tag names work; the backend reads them from the config.

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pipe" background="#FF0000"/>
  </RectangleLabels>
</View>
```

## Commands

```powershell
uv run lb-anythings serve [--host H] [--port P]   # the ML backend
uv run lb-anythings train                         # one training run, here, now
uv run lb-anythings mine [--video V] [...]        # write the hardest frames of a video
```

`train` is also what the server spawns for you when the retrain threshold trips or you press
Start Training. Run it by hand to retrain on demand.

## Settings

Every setting is read once at startup from the environment or a `.env` file in the working
directory. **A shell variable beats `.env`.** The effective values are logged on the first
line of every run, with the access token masked, so you can always see what actually loaded.

| Variable | Default | Meaning |
|---|---|---|
| `LB_HOST` | `0.0.0.0` | bind address |
| `LB_PORT` | `9090` | port |
| `LB_LOG_LEVEL` | `INFO` | log level |
| `LB_DATA_DIR` | `data` | everything the backend writes lives here |
| `LB_CHECKPOINT` | unset | a checkpoint to serve; the newer of it and the trained one wins |
| `LB_CONF` | `0.25` | confidence floor for predictions |
| `LB_IMGSZ` | `1024` | inference and training image size |
| `LB_TRAIN_RUN_NAME` | `active` | name of the training run's output folder |
| `LB_RETRAIN_EVERY` | `25` | retrain threshold: train every N examples |
| `LB_MIN_EXAMPLES` | `4` | never train on fewer than this |
| `LB_TRAIN_BASE_MODEL` | `yolo11s.pt` | base weights to train from |
| `LB_TRAIN_EPOCHS` | `100` | epochs |
| `LB_TRAIN_PATIENCE` | `30` | early-stop patience |
| `LB_TRAIN_BATCH` | `8` | batch size |
| `LB_TRAIN_LR0` | unset | initial learning rate, when you want to set it |
| `LB_DEVICE` | `0` | training device |
| `LB_MINE_VIDEO` | unset | video to mine |
| `LB_MINE_STRIDE` | `15` | score every Nth frame |
| `LB_MINE_TOPN` | `40` | how many frames to keep |
| `LB_MINE_GAP` | `60` | fewest frames between two picks |
| `LB_MINE_UNCERTAIN_LO` | `0.25` | uncertainty band floor |
| `LB_MINE_UNCERTAIN_HI` | `0.55` | uncertainty band ceiling |
| `LB_MINE_CONF` | `0.15` | confidence floor while mining, below the prediction one |
| `LB_MINE_OUT` | `<data dir>/hard_frames` | where mined frames go |
| `LABEL_STUDIO_URL` | unset | Label Studio's base URL (`LABEL_STUDIO_HOSTNAME` also works) |
| `LABEL_STUDIO_API_KEY` | unset | Label Studio access token |

Under the data directory:

```
data/
├── examples/                images, one label file each, and classes.txt: the training set
├── runs/active/             the training run's output, including weights/best.pt
├── runs/active.json, .log   what the current run is, and what it printed
├── dataset/                 rebuilt from the training set on every run
├── cache/                   images fetched from Label Studio
└── hard_frames/             what `mine` writes
```

## Coming from the old `ls_backend`

The Label Studio project needs no change: the same Connect Model URL works. Bring the
examples and the checkpoint across once:

```powershell
$data = "C:\path\to\LB-anythings\data"
New-Item -ItemType Directory -Force "$data\examples" | Out-Null
Copy-Item -Recurse "<old repo>\ls_data\images" "$data\examples\images"
Copy-Item -Recurse "<old repo>\ls_data\labels" "$data\examples\labels"
Set-Content "$data\examples\classes.txt" "pipe"    # one label per line, in class-index order
New-Item -ItemType Directory -Force "$data\runs\active\weights" | Out-Null
Copy-Item "<old repo>\runs\pipe_ls\weights\best.pt" "$data\runs\active\weights\best.pt"
```

The old layout is read as-is: the same `task<id>` file names and the same normalized lines.
The only new file is `classes.txt`, which names the classes the label indices refer to.

The environment variables were renamed:

| Old | New |
|---|---|
| `PIPE_WEIGHTS` | `LB_CHECKPOINT` |
| `PIPE_CONF` | `LB_CONF` |
| `PIPE_IMGSZ` | `LB_IMGSZ` |
| `PIPE_TRAIN_EVERY` | `LB_RETRAIN_EVERY` |
| `PIPE_TRAIN_MODEL` | `LB_TRAIN_BASE_MODEL` |
| `PIPE_TRAIN_EPOCHS` | `LB_TRAIN_EPOCHS` |
| `PIPE_TRAIN_PATIENCE` | `LB_TRAIN_PATIENCE` |
| `PIPE_TRAIN_BATCH` | `LB_TRAIN_BATCH` |
| `PIPE_LR0` | `LB_TRAIN_LR0` |
| `PIPE_DEVICE` | `LB_DEVICE` |
| `PIPE_MINE_VIDEO`, `_STRIDE`, `_TOPN`, `_GAP`, `_OUT` | `LB_MINE_VIDEO`, `_STRIDE`, `_TOPN`, `_GAP`, `_OUT` |
| `PIPE_BACKEND` | gone: the colour detector is not ported |

Two behaviours changed on purpose. A shell variable now beats `.env`, which is the convention
everywhere else. And `/is_training` tells the truth, so a second training run cannot start on
top of one already going.

## Adding a detector

Nothing in the core knows about YOLO. A detector is anything that turns an image into
detections:

```python
class Detector(Protocol):
    @property
    def version(self) -> str: ...
    def detect(self, image: Image) -> Sequence[Detection]: ...
```

Implement it, and a `DetectorFactory` that loads one from a checkpoint, both under
`src/lb_anythings/adapters/outbound/<yours>/`. Then name it in `production_ports` in
`src/lb_anythings/bootstrap/container.py`. Nothing else changes: predictions, collection and
the retrain loop are all written against the port. The same is true of the example store, the
trainer, the media resolver and the frame source.

## Working on it

```powershell
uv sync --no-default-groups --group dev   # no ML stack: the tests do not need it
uv run pytest -q
uv run mypy
uv run ruff check .
uv run lint-imports                       # the dependency rule, enforced
```

The code is a hexagon: `domain` (pure rules), `application` (use cases and the ports they
need), `adapters` (inbound HTTP and CLI; outbound YOLO, filesystem, Label Studio, OpenCV,
subprocess, background), `bootstrap` (settings and the composition root). Dependencies only
ever point inward, and `lint-imports` fails the build when they do not.

The vocabulary is in [CONTEXT.md](CONTEXT.md), the decisions worth keeping in
[docs/adr/](docs/adr/), and the whole design in
[docs/specs/0001-hexagonal-label-studio-backend.md](docs/specs/0001-hexagonal-label-studio-backend.md).

The real detector's test runs only when you point it at a checkpoint and a frame:

```powershell
$env:LB_TEST_CHECKPOINT = "C:\path\to\best.pt"
$env:LB_TEST_IMAGE = "C:\path\to\frame.jpg"
uv run pytest -m gpu
```

## The HTTP API

What Label Studio calls, and what you can call yourself:

| Endpoint | What it does |
|---|---|
| `GET /health`, `GET /` | `{"status": "UP", "model_version": ...}` |
| `POST /setup` | takes the label config; returns the serving version |
| `POST /predict` | takes tasks; returns one prediction each, in order |
| `POST /webhook` | annotation events and the Start Training button |
| `POST /train` | what older Label Studio versions call for Start Training |
| `GET /is_training` | whether a training run is active |
| `GET /metrics`, `POST /versions` | for older Label Studio versions |
