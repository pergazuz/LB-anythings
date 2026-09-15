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
  Cancelled and skipped annotations are dropped; one with no boxes is kept, as a negative
  example, but does not count toward the retrain threshold.
- **Retrains.** When the training set has grown enough since the last run, a training run
  starts on its own and the next prediction uses the checkpoint it produced. No restart.
  "Enough" scales: 25 new examples while the set is small, 10% of it once it is large.
- **Trains on demand.** The Start Training button pulls every annotated task from the project
  and trains on all of them, including annotations made before the backend was connected.
- **Mines.** `lb-anythings mine` finds the frames of a video the detector is least sure about,
  so you label its blind spots instead of random frames.
- **Records.** Every training run is logged to MLflow with its metrics, its settings and a copy
  of the checkpoint it produced, so you can see whether labelling is still paying off and go
  back to a better checkpoint.

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
- **Start model training on annotation submission:** on. This is the toggle that makes Label
  Studio send the annotation events. With it off nothing is collected and nothing ever
  retrains, however many tasks you label.

**Validate and Save** calls `GET /health` then `POST /setup`. Open a task and the boxes appear.
The model card that appears under **Settings → Model** is also where the **Start Training**
button lives.

For the backend to fetch images you uploaded to Label Studio, it needs to reach Label Studio
back. Label Studio sends a hostname and access token with most requests; when it does not, set
them yourself:

```powershell
$env:LABEL_STUDIO_URL = "http://localhost:8080"
$env:LABEL_STUDIO_API_KEY = "<your access token>"   # Account & Settings -> Access Token
```

These are also what the **Start Training** button needs.

Use a **personal access token** (Account & Settings → Access Token). Label Studio 1.23 disables
legacy tokens by default, and a personal access token is a JWT: the backend exchanges it for a
short-lived one and sends it as a bearer token. A legacy token still works on an instance that
still allows them.

Label Studio hands the backend its own credentials at setup and those are preferred, but 1.23
sends a *legacy* token even when legacy tokens are disabled — and then rejects it. When that
happens the backend falls back to the token you configured here, for the same host only, and
says so in the log. Without `LABEL_STUDIO_API_KEY` set there is nothing to fall back to and
images will not load.

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
Start Training: it *is* the training run, so it does not itself check whether another one is
going. Run it by hand to retrain on demand, but check `GET /is_training` first: starting one
on top of a run the server launched double-books the GPU, and both write the same checkpoint.

### Mining

`mine` needs a checkpoint: it asks the current detector what it is unsure about, so with none
yet there is nothing to ask, and it says so and stops. It scores every `LB_MINE_STRIDE`-th
frame, keeps the `LB_MINE_TOPN` most uncertain, spreads the picks at least `LB_MINE_GAP` frames
apart, and writes them as `hard_<frame index>_s<score>.jpg` so a labelled frame is traceable
back to the video:

```powershell
$env:LB_MINE_VIDEO = "C:\path\to\clip.mov"
uv run lb-anythings mine
# wrote 40 Hard Frames -> data\hard_frames
```

Then import `data\hard_frames` into Label Studio as a new set of tasks and label those next.
That is the point of it: they are the frames the detector is worst at.

## What each training run recorded

Every run is logged to MLflow, on this machine, into a SQLite file. Nothing leaves the machine
and there is no account to create. To look at it:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
# then open http://127.0.0.1:5000
```

A run carries every training setting as a parameter, box/class/DFL loss and mAP50 and mAP50-95
per epoch as metrics, and `best.pt` and `last.pt` as artifacts. Alongside those it records what
only the backend knows at the moment it launched the run:

| Parameter | Meaning |
|---|---|
| `trigger` | `retrain-threshold`, `start-training` or `command-line` |
| `training_set_size` | how many positive examples existed when the run launched |
| `serving_version` | the checkpoint this run was trying to beat |
| `exported_tasks`, `unannotated_tasks`, `uncollected_tasks` | Start Training only: what the project export held |

With `LB_TRACKING_SYSTEM_METRICS` on, a run also carries CPU, memory and GPU utilisation
sampled while it trained — MLflow leaves that off by default, and training is the one thing
here that loads the card.

Two tabs in MLflow's run view stay empty, and should: **Model metrics** lists metrics attached
to a *logged model*, and nothing here calls `mlflow.log_model` — the checkpoint is archived as
an artifact instead, which is what you want to roll back to. **Traces** is for instrumenting
LLM calls and has nothing to do with training. The numbers you want are the run's own metrics:
open a run and look at **Overview → Metrics**, or use the experiment's **Chart** view to
compare runs.

### How often it retrains

The threshold is a floor, not the whole rule. A run starts when the training set has grown,
since the last run, by `LB_RETRAIN_EVERY` examples **or** by `LB_RETRAIN_GROWTH` of its own
size — whichever is larger:

| Training set | Next run at | Step |
|---|---|---|
| 50 | 75 | 25 (the floor) |
| 250 | 275 | 25 (the floor) |
| 500 | 550 | 50 |
| 1000 | 1100 | 100 |
| 5000 | 5500 | 500 |

Accuracy improves as a power law in the number of examples, so equal *relative* growth buys
roughly equal accuracy while equal *absolute* growth buys less and less. Twenty-five more
examples doubles a set of 25 and is a rounding error on a set of 1000, but costs the same GPU
minutes either way. `LB_RETRAIN_GROWTH=0` gives you a fixed step back. The reasoning and the
sources are in [ADR 0004](docs/adr/0004-the-retrain-threshold-grows-with-the-training-set.md).

`training_set_size` is the one to plot against. Detector quality on its own says little; mAP
against how much you have labelled is what tells you whether labelling is still paying off, or
whether the curve has flattened and your time is better spent elsewhere. Both halves land on
one row because the backend opens the run, hands its id to the training run, and MLflow
resumes it. That last part matters beyond
curiosity: a training run overwrites `runs/active/weights/best.pt`, so **the archived copy is
the only way back to an earlier checkpoint**. Runs are named `<run name>-<UTC stamp>`, because
the folder they train into is always the same one.

The cost is disk. A checkpoint here is about 19 MB and each run archives two of them, so a
retrain every 25 examples runs to roughly 40 MB per run, a couple of GB after fifty. Prune
`data\mlflow` when it gets big, or point `LB_TRACKING_URI` at a drive where you do not care.

Tracking needs the `tracking` dependency group, which a plain `uv sync` installs. With it
absent the trainer behaves exactly as it would otherwise and says nothing about it.

`LB_TRACKING_URI` takes any MLflow tracking URI, so pointing several machines at one server is
`http://mlflow.local:5000`. It defaults to SQLite because MLflow has deprecated the plain
directory store, which now refuses to open at all.

## Settings

Every setting is read once at startup from the environment or a `.env` file in the working
directory. **A shell variable beats `.env`.** The effective values are logged at INFO on the
first line of every run, with the access token masked, so you can always see what loaded.
Paths are logged resolved, so that line says where the backend will really write.

| Variable | Default | Meaning |
|---|---|---|
| `LB_HOST` | `0.0.0.0` | bind address |
| `LB_PORT` | `9090` | port |
| `LB_LOG_LEVEL` | `INFO` | log level |
| `LB_DATA_DIR` | `data` | everything the backend writes lives here; a relative path is anchored at startup to the directory you start in |
| `LB_CHECKPOINT` | unset | a checkpoint to serve; the newer of it and the trained one wins |
| `LB_CONF` | `0.25` | confidence floor for predictions |
| `LB_IMGSZ` | `1024` | inference and training image size |
| `LB_TRAIN_RUN_NAME` | `active` | name of the training run's output folder |
| `LB_RETRAIN_EVERY` | `25` | retrain threshold: fewest new examples worth a run |
| `LB_RETRAIN_GROWTH` | `0.10` | ...and never on less than this share of the training set |
| `LB_MIN_EXAMPLES` | `4` | never train on fewer than this |
| `LB_TRAIN_BASE_MODEL` | `yolo11s.pt` | base checkpoint to train from |
| `LB_TRAIN_EPOCHS` | `100` | epochs |
| `LB_TRAIN_PATIENCE` | `30` | early-stop patience |
| `LB_TRAIN_BATCH` | `8` | batch size |
| `LB_TRAIN_LR0` | unset | initial learning rate, when you want to set it |
| `LB_DEVICE` | `0` | training device |
| `LB_TRACKING` | `true` | record training runs to MLflow |
| `LB_TRACKING_URI` | SQLite in `<data dir>/mlflow` | MLflow tracking URI; a server URL also works |
| `LB_TRACKING_EXPERIMENT` | `lb-anythings` | the MLflow experiment to record under |
| `LB_TRACKING_SYSTEM_METRICS` | `true` | also sample CPU, memory and GPU during a run |
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
├── examples/images/         the training set's images
├── examples/labels/         one label file per image
├── examples/classes.txt     the class names the label indices refer to
├── runs/active/             the training run's output, including weights/best.pt
├── runs/active.json, .log   what the current run is, and what it printed
├── dataset/                 rebuilt from the training set on every run
├── mlflow/mlflow.db         what each training run recorded
├── mlflow/artifacts/        each run's archived checkpoints
├── cache/                   images fetched from Label Studio
└── hard_frames/             what `mine` writes
```

## Coming from the old `ls_backend`

The Label Studio project needs no change: the same Connect Model URL works. Bring the
examples and the checkpoint across once:

```powershell
$data = "C:\path\to\LB-anythings\data"
New-Item -ItemType Directory -Force "$data\examples\images", "$data\examples\labels" | Out-Null
Copy-Item -Recurse -Force "<old repo>\ls_data\images\*" "$data\examples\images"
Copy-Item -Recurse -Force "<old repo>\ls_data\labels\*" "$data\examples\labels"
Set-Content "$data\examples\classes.txt" "pipe"    # one label per line, in class-index order
New-Item -ItemType Directory -Force "$data\runs\active\weights" | Out-Null
Copy-Item -Force "<old repo>\runs\pipe_ls\weights\best.pt" "$data\runs\active\weights\best.pt"
```

The examples you bring across count, and they train immediately: with no previous run to
measure growth against, the first training run starts as soon as `LB_MIN_EXAMPLES` is met
rather than waiting for more annotations.

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

Four behaviours changed on purpose. A shell variable now beats `.env`, which is the convention
everywhere else. `/is_training` tells the truth, so neither the retrain threshold nor Start
Training can begin a run on top of one already going. One unreadable image no longer fails the
whole predict batch: that task gets an empty prediction and the rest are served. And cancelled
annotations are no longer stored as examples, so a skip cannot teach the detector anything.

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
`src/lb_anythings/bootstrap/container.py`, and in `mine_with_yolo` in the same file, which
builds its own factory for the CLI. Nothing else changes: predictions, collection and the
retrain loop are all written against the port. The same is true of the example store, the
trainer, the media resolver and the frame source.

## Working on it

```powershell
uv sync --no-default-groups --group dev   # no ML stack, no MLflow: the tests need neither
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

The real detector's test runs only when you point it at a checkpoint and a frame, and only
with the ML stack installed: after a `--no-default-groups` sync it skips. Run a plain `uv sync`
first:

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
