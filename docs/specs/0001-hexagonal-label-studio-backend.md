# LB-anythings: a detector-agnostic Label Studio ML backend, rewritten hexagonally

Rewrite of the `ls_backend` folder of the pipe-counting project (four scripts, about
650 lines) into this repo as a hexagonal service. Vocabulary is defined in
`CONTEXT.md`; decisions with a "why" worth keeping are in `docs/adr/`. Triage:
`ready-for-agent`.

## Problem Statement

I run a Label Studio project where I correct predicted boxes instead of drawing them,
and the backend retrains itself as I label. The backend I have works, but:

- It is welded to one project. "pipe" is in file names, run names and defaults, and
  the second detector reaches into a sibling package, so I cannot point it at a
  different detection project without editing code.
- Nothing in it can be tested without a GPU, trained weights and a running Label
  Studio, so every change is verified by hand.
- Configuration is read from the environment at import time in three different
  files, so it is easy to be running values other than the ones I think I set.
- Training is fire-and-forget: the backend always reports "not training", and if the
  threshold trips during a run a second run starts on the same GPU.
- The detector class inherits from a framework base class for the sake of two helper
  functions, and that base class drags in a Flask server I no longer use.

## Solution

A new service in this repo that speaks exactly the same wire protocol to Label
Studio, so the existing project reconnects with no changes, but is built as a
hexagon:

- A pure core that knows about Tasks, Annotation Targets, Predictions, Annotations,
  Detections, Examples, the Training Set, Training Runs, Checkpoints and Hard
  Frames, and nothing about YOLO, HTTP, files or Label Studio's API.
- Six use cases: set up for a project, predict for Tasks, ingest an Annotation,
  collect the whole project and train, report status, mine Hard Frames.
- Ports for everything that touches the world: detector, checkpoint repository,
  example store, trainer, task media resolver, Label Studio project client, frame
  source, background runner.
- Adapters: FastAPI and a CLI inbound; YOLO, filesystem, subprocess, Label Studio
  HTTP and OpenCV outbound.
- One composition root that reads settings once and wires adapters to use cases.
- Tests that drive the whole thing through HTTP with in-memory fakes, plus a
  contract test per real adapter.

The Operator can then reuse the same service for any single-class or multi-class
bounding-box project by pointing it at a different Checkpoint and data directory.

## User Stories

### Connecting and setup

1. As an Operator, I want to connect the backend from Label Studio's Connect Model
   dialog with only its URL, so that Validate and Save passes on the first try.
2. As an Operator, I want the backend to read the control tag, image tag, image
   field and label values from whatever label config the project has, so that I
   never hard-code tag names.
3. As an Operator, I want setup to fail with a readable reason when the label config
   does not contain exactly one RectangleLabels control bound to exactly one Image,
   so that a misconfigured project is caught at connect time rather than at the
   first Prediction.
4. As an Operator, I want a later setup with a changed label config to take effect
   without a restart, so that editing the labeling interface never requires touching
   the server.
5. As an Operator, I want the health endpoint to name the Checkpoint currently
   serving, so that I can tell at a glance which model is live.
6. As an Operator, I want the backend to start and pass validation when no
   Checkpoint exists yet, so that I can begin a brand-new project by labelling from
   scratch and let the first Training Run create the Detector.

### Predicting

7. As an Annotator, I want predicted boxes to appear when I open a Task, so that I
   correct instead of draw.
8. As an Annotator, I want every predicted box to carry the Detector's confidence,
   so that Label Studio can display and sort by it.
9. As an Annotator, I want each Prediction's overall score to be the mean confidence
   of its boxes, and zero when there are none, so that Label Studio's ordering by
   prediction score stays meaningful.
10. As an Annotator, I want each predicted box to already carry the right label
    value from my project, so that I do not have to reclassify boxes the Detector
    found.
11. As an Annotator, I want Predictions to work for images I uploaded directly to
    Label Studio, not only for externally hosted URLs, so that I do not need a
    separate file server.
12. As an Annotator, I want Predictions to work for images served from Label
    Studio's local-files storage, so that the folder-import workflow is supported.
13. As an Operator, I want one Task whose image cannot be fetched or decoded to
    yield an empty Prediction and a logged warning, so that a single bad Task does
    not fail the whole batch Label Studio asked for.
14. As an Operator, I want the Detector to pick up a newer Checkpoint produced by a
    Training Run before the next Prediction, so that Predictions improve as I label
    without a restart.
15. As an Operator, I want a reload to be safe while other Predictions are in
    flight, so that concurrent requests from Label Studio never see a half-loaded
    Detector.
16. As an Operator, I want the Detector's confidence floor and inference image size
    to be settings, so that I can tune recall against noise per project.
17. As an Operator, I want Label Studio's force-reload flag to reload the Detector
    from the latest Checkpoint, so that I can swap weights by hand and pick them up
    on demand.

### Collecting Examples

18. As an Annotator, I want every Annotation I submit or update to become an
    Example, so that my corrections accumulate into a Training Set.
19. As an Operator, I want a second Annotation of the same Task to replace the
    earlier Example, so that the Training Set holds one Example per Task and never
    duplicates an image.
20. As an Operator, I want cancelled or skipped Annotations to be ignored, so that
    skipped Tasks do not pollute the Training Set.
21. As an Operator, I want an Annotation with no boxes to be kept as an Example with
    no boxes but not counted toward the Retrain Threshold, so that I can record
    negatives without them triggering training on their own.
22. As an Operator, I want the webhook to acknowledge immediately and do its work in
    the background, so that submitting an Annotation in Label Studio never stalls on
    the backend.
23. As an Operator, I want a webhook that arrives before setup has happened to be
    acknowledged and skipped with a reason, so that Label Studio does not retry
    endlessly and I can see in the log why nothing was stored.
24. As an Operator, I want an Example's boxes stored independent of the image's
    pixel size, so that the same Example stays valid whatever resolution the trainer
    uses.

### Automatic training

25. As an Operator, I want a Training Run to start on its own every time the
    Training Set grows by the Retrain Threshold, so that the model improves while I
    label.
26. As an Operator, I want the Retrain Threshold to be a setting, so that I can
    trade training frequency against GPU time.
27. As an Operator, I want the automatic trigger to refuse to start a second
    Training Run while one is active, so that the GPU is never double-booked and the
    Checkpoint is never written by two processes.
28. As an Operator, I want the automatic trigger to log its reason when it declines
    to train, so that I know whether the set was too small, a run was already
    active, or something else happened.

### Manual training

29. As an Operator, I want the Start Training button to pull every annotated Task
    from the project through Label Studio's API and store each as an Example, so
    that I train on everything I have labelled, including Annotations made before
    the backend was connected.
30. As an Operator, I want Start Training to refuse with a clear reason when the
    Label Studio URL, API key or project id is missing, so that I know exactly which
    setting to add.
31. As an Operator, I want Start Training to report how many Examples it collected
    and whether a Training Run launched, so that the backend log tells me what
    happened.
32. As an Operator, I want a minimum Training Set size before any Training Run
    launches, so that training never starts on a set too small to split.
33. As an Operator, I want to launch the same Training Run from the shell, so that I
    can retrain without going through Label Studio.
34. As an Operator, I want the older `/train` endpoint to behave like Start Training,
    so that older Label Studio versions still work.

### Training Runs

35. As an Operator, I want `/is_training` to tell the truth, so that Label Studio's
    model card reflects reality.
36. As an Operator, I want a Training Run whose process died without producing a
    Checkpoint to be recognised as failed, so that a stale "training" state never
    blocks future runs.
37. As an Operator, I want a Training Run to execute in its own process, detached
    from the server, so that a training crash and a server restart cannot take each
    other down.
38. As an Operator, I want a Training Run to build its train/validation split fresh
    from the Training Set each time, so that Examples added since the last run are
    always included.
39. As an Operator, I want the trainer's base model, epochs, patience, batch size,
    image size, device and optional learning rate to be settings, so that I can
    experiment without editing code.
40. As an Operator, I want the Training Run to write its Checkpoint where the
    Detector looks for the latest one, so that the loop from label to better
    Prediction closes on its own.
41. As an Operator, I want the trainer to be Windows-safe, so that it runs on the
    machine I actually have.

### Mining Hard Frames

42. As an Operator, I want to mine the N frames of a video the Detector is least
    sure about, so that I label the model's weak spots instead of random frames.
43. As an Operator, I want the mined frames spread out in time by a minimum gap, so
    that I do not label forty near-identical frames.
44. As an Operator, I want small uncertain boxes to weigh more than large ones, so
    that the known weakness on distant, small objects is targeted first.
45. As an Operator, I want the video path, sampling stride, top-N, minimum gap,
    uncertainty band, confidence floor and output folder to be settings or flags, so
    that I can tune mining per video.
46. As an Operator, I want mined frames named with their frame index and score, so
    that I can trace any labelled frame back to the video.
47. As an Operator, I want mining to use the newest Checkpoint by default, so that
    it targets the current model's weaknesses rather than an old one's.
48. As an Operator, I want mining to print progress and a final count, so that a
    long scan is visibly alive.

### Configuration and operations

49. As an Operator, I want every setting read once at startup from the shell or a
    `.env` file, with the shell winning, so that the backend behaves like every
    other tool I run.
50. As an Operator, I want the effective settings logged at startup with secrets
    redacted, so that I can see what actually loaded.
51. As an Operator, I want to point the backend at any Checkpoint and any data
    directory, so that I can reuse it for a different detection project or keep
    using my existing Examples.
52. As an Operator, I want the API key never to appear in logs or responses, so that
    a shared log does not leak credentials.
53. As an Operator, I want the compatibility endpoints `/metrics` and `/versions` to
    keep answering, so that older Label Studio versions validate.
54. As an Operator, I want a README that takes me from clone to connected in a
    handful of commands, including the Docker host note, so that setup takes
    minutes.
55. As an Operator, I want a documented one-time step that brings my 333 existing
    Examples and current Checkpoint into the new data directory, so that I do not
    start from zero.

### Development

56. As a developer, I want the core testable with no GPU, no torch, no files and no
    Label Studio, so that the test suite runs in seconds anywhere.
57. As a developer, I want the dependency direction enforced by a linter that fails
    the build, so that the architecture cannot erode quietly.
58. As a developer, I want to add a new Detector by implementing one interface and
    registering it in the composition root, so that extension never touches the
    core.
59. As a developer, I want to swap the example store, trainer or media resolver the
    same way, so that each edge of the system is independently replaceable.
60. As a developer, I want the vocabulary in CONTEXT.md and the non-obvious
    decisions in ADRs, so that the next person or agent does not re-litigate them.
61. As a developer, I want the GPU-dependent adapter test to run only when a
    Checkpoint path is provided, so that CI passes without a GPU and I can still
    verify the real Detector locally.

## Implementation Decisions

### Repository and toolchain

- Python 3.11 managed by uv, `src` layout, one package named `lb_anythings`,
  hatchling build backend. Everything runs through `uv run`.
- Runtime dependencies: fastapi, uvicorn, pydantic v2, pydantic-settings, httpx,
  numpy, opencv-python, ultralytics, pyyaml, psutil. Dev dependencies: pytest,
  ruff, import-linter. `label-studio-ml` is not a dependency (ADR 0001).
- torch and torchvision are pinned to the PyTorch cu128 index exactly as the parent
  project does, because the target machine's GPU is Blackwell-class and the default
  PyPI wheels do not support it. The README repeats the parent's warning: never
  reinstall torch from the default index.
- One console entry point, `lb-anythings`, with three subcommands: `serve`, `train`,
  `mine`. `train` is both the Operator's shell command and what the server spawns.

### Architecture and the dependency rule

- Four rings, each a subpackage: `domain` (entities, value objects, pure rules and
  typed errors), `application` (ports declared as Protocols, use cases, and the
  detector cache service), `adapters` (inbound: `http`, `cli`; outbound: `yolo`,
  `filesystem`, `subprocess`, `labelstudio`, `opencv`, `background`), and `bootstrap` (settings
  and the composition root).
- import-linter contracts, run in CI and as a pytest test: `domain` imports only the
  standard library; `application` imports `domain` (and numpy, for the image type
  that crosses ports); each outbound adapter imports `application` and `domain` and
  never another adapter; inbound adapters import `application` and `domain`; only
  `bootstrap` imports adapters.
- No framework base classes anywhere in `domain` or `application`.

### Domain model

- Coordinates. The canonical box is normalized `xyxy` in the unit square, relative
  to its image. Conversions to Label Studio's percent `x, y, width, height` and to
  YOLO's normalized `cx, cy, w, h` are pure functions in `domain` and the only
  places those formats exist.
- Detection: a normalized box, a score in [0, 1], and a label string (the Detector's
  own class name).
- Prediction: the regions for one Task, an overall score equal to the mean of the
  region scores (zero when empty), and the Checkpoint version that produced it.
- Annotation Target: control name, object name, image field key (with the leading
  `$` removed), and the ordered label values. Parsed from label-config XML by a
  domain parser. Exactly one RectangleLabels bound (via `toName`) to exactly one
  Image is required; anything else raises `InvalidLabelConfig` with a message that
  names what was found.
- Label mapping rule: a Detection's label maps to the Annotation Target label with
  the same name when one exists, otherwise to the first target label. A single-class
  Detector therefore always lands on the project's label regardless of its class
  name.
- Example: task id and zero or more Ground-Truth Boxes (box plus the Annotator's label).
  The image is stored beside it by the Example Store. Identity is the task id; saving
  again replaces. Zero boxes is a valid negative Example. An Annotator's label is kept
  as given; only a missing label takes the project's first.
- Training Set size is the number of Examples with at least one box; negatives are
  stored but not counted, matching the previous backend.
- Retrain rule, evaluated after each Example is saved: launch when the size is at
  least the minimum, is a positive multiple of the Retrain Threshold, and no
  Training Run is active. Every refusal carries a reason.
- Training Run: id, start time, status (`running`, `succeeded`, `failed`), and on
  success the Checkpoint it produced. Status rule: running while its process is
  alive; succeeded once a Checkpoint newer than its start time exists in the run's
  output; failed if the process is gone and no such Checkpoint exists.
- Checkpoint: a path and a modified time; its version string is its file name plus its
  modification time, so every Training Run's output is a distinct version to Label Studio
  and a reload is visible in `model_version`. The
  latest Checkpoint is the newer, by modified time, of the trainer's output for the
  configured run name and the explicitly configured Checkpoint (ties go to the
  trained one), else none. An Operator who drops in a fresher file is served it.
- Hard Frame scoring, a pure function over a frame's Detections: for each Detection
  whose score lies in the uncertainty band `[lo, hi)`, add 1, plus 2 if its area is
  under 2% of the frame, or plus 1 if under 5%. Selection: sort frames by score
  descending, greedily keep a frame only if it is at least `gap` frames from every
  frame already kept, stop at top-N, and return the picks in time order.

### Ports (declared in `application`)

- `Detector`: detect(image) returns Detections; exposes its version string.
- `DetectorFactory`: load(checkpoint) returns a Detector. The application also owns
  a `NullDetector` that returns no Detections and reports version `none`, used when
  no Checkpoint exists.
- `CheckpointRepository`: latest() returns the latest Checkpoint or none.
- `ExampleStore`: save(example, image); positive_count(); all(). The image travels beside
  the Example as an application-level value: the domain holds no pixels.
- `Trainer`: start() returns a Training Run or raises `TrainingAlreadyActive`;
  active() returns the active run or none; refresh(run) returns the run with its
  current status.
- `TaskMediaResolver`: load(image reference, credentials) returns a decoded image or
  raises `MediaUnavailable`. Images cross ports as decoded arrays with known width
  and height; downloading, caching and decoding live in adapters.
- `LabelStudioProjectClient`: annotated_tasks(project id, credentials) returns each
  Task with its first non-cancelled Annotation's regions.
- `FrameSource`: opened on a video; exposes frame count and frame size; iterates
  (index, image) at a stride; reads a specific index.
- `BackgroundRunner`: run(callable). A thread in production; synchronous in tests.

### Use cases

- SetupProject(label config, optional hostname, optional access token): parses the
  Annotation Target, replaces the in-memory project context (target plus
  credentials), returns the current version. Credentials given here take precedence
  over settings for media resolution and project export, as before.
- PredictTasks(tasks, optional label config, force reload): establishes the project
  context if a label config is supplied and none exists; asks the detector cache for
  the current Detector (which compares the latest Checkpoint with the loaded one and
  reloads under a lock when newer, or when forced); for each Task loads the image,
  detects, maps labels, converts to Label Studio percent regions. A Task whose image
  fails to load yields an empty Prediction and a warning; the batch succeeds.
- IngestAnnotation(event): if no project context and the event carries no usable label
  config, returns skipped with a reason (events carry the project's config, which
  establishes the context the same way a predict request's does); ignores cancelled or
  skipped Annotations; builds an Example from the Annotation's
  rectangle regions only; loads the image through the resolver and saves the
  Example; applies the retrain rule and starts the Trainer if due. Returns what it
  did (stored, box count, set size, training launched or the refusal reason).
- TrainOnProject(project id): requires a Label Studio URL and API key (from setup or
  settings) and a project id, else `MissingLabelStudioCredentials`; pulls annotated
  Tasks; saves each as an Example; launches a Training Run if the set meets the
  minimum and none is active. Returns the collected count and the training outcome.
- ReportStatus(): current version, whether a Training Run is active (after
  refreshing its status), and the list of known versions.
- MineHardFrames(parameters): iterates the FrameSource at the stride, detects with
  the confidence floor, scores each frame, selects the picks, and returns them with
  their images. The CLI adapter writes the files.

### HTTP contract (unchanged from the previous backend, Label Studio compatible)

- `GET /` and `GET /health` return `status: "UP"` and `model_version`.
- `POST /setup` accepts `schema` or `label_config`, optional `hostname`,
  `access_token`, `force_reload`; returns `model_version`. An invalid label config
  is a 400 whose detail is the parser's message.
- `POST /predict` accepts `tasks`, optional `label_config`, optional `force_reload`;
  returns `results` (one Prediction per Task, in order) and `model_version`. With no
  project context and no label config it returns empty results and a null version.
- `POST /webhook` accepts `action` plus the event payload. `ANNOTATION_CREATED` and
  `ANNOTATION_UPDATED` schedule IngestAnnotation; `START_TRAINING` schedules TrainOnProject with the project id from the payload; other
  actions are logged and skipped. Always 201: `job_id` when scheduled, or `status: "skipped"`
  with a `reason` otherwise (no project context, an unhandled action, an unusable config).
- `POST /train` behaves exactly like a `START_TRAINING` webhook.
- `GET /is_training` returns the truthful boolean.
- `GET /metrics` returns an empty object. `POST /versions` returns the known
  versions, empty when none.
- Request and response shapes are pydantic models in the HTTP adapter and tolerate
  extra fields, because Label Studio's payloads vary by version.

### CLI

- `serve` takes host and port (defaults `0.0.0.0` and `9090`) and runs uvicorn.
- `train` runs one Training Run in the current process and exits; it prints the
  split sizes and the resulting Checkpoint path.
- `mine` takes the mining parameters as flags that override settings, and writes
  `hard_<frame index, zero-padded to six digits>_s<integer score>.jpg` files to the
  output folder.

### Settings

- One pydantic-settings model, read once in `bootstrap`, from the shell and a `.env`
  in the working directory, with the shell winning. This deliberately reverses the
  previous backend's `.env`-wins behaviour; the startup log of effective values is
  what makes stale shell variables visible.
- Prefix `LB_` for the backend's own knobs. Label Studio's conventional names are
  kept as-is.

  | name | default | meaning |
  |---|---|---|
  | `LB_DATA_DIR` | `./data` | root of Examples, runs, cache and mined frames |
  | `LB_CHECKPOINT` | unset | explicit Checkpoint to serve when no trained one is newer |
  | `LB_CONF` | `0.25` | Detector confidence floor for Predictions |
  | `LB_IMGSZ` | `1024` | inference and training image size |
  | `LB_RETRAIN_EVERY` | `25` | Retrain Threshold |
  | `LB_MIN_EXAMPLES` | `4` | minimum Training Set size for any Training Run |
  | `LB_TRAIN_RUN_NAME` | `active` | name of the trainer's output run |
  | `LB_TRAIN_BASE_MODEL` | `yolo11s.pt` | base weights for training |
  | `LB_TRAIN_EPOCHS` | `100` | epochs |
  | `LB_TRAIN_PATIENCE` | `30` | early-stop patience |
  | `LB_TRAIN_BATCH` | `8` | batch size |
  | `LB_TRAIN_LR0` | unset | optional explicit initial learning rate |
  | `LB_DEVICE` | `0` | training device |
  | `LB_MINE_VIDEO` | unset | video to mine |
  | `LB_MINE_STRIDE` | `15` | sample every Nth frame |
  | `LB_MINE_TOPN` | `40` | frames to keep |
  | `LB_MINE_GAP` | `60` | minimum frames between picks |
  | `LB_MINE_UNCERTAIN_LO` | `0.25` | uncertainty band lower bound |
  | `LB_MINE_UNCERTAIN_HI` | `0.55` | uncertainty band upper bound |
  | `LB_MINE_CONF` | `0.15` | Detector confidence floor while mining |
  | `LB_MINE_OUT` | `<data dir>/hard_frames` | output folder |
  | `LB_HOST`, `LB_PORT` | `0.0.0.0`, `9090` | serve defaults |
  | `LB_LOG_LEVEL` | `INFO` | log level |
  | `LABEL_STUDIO_URL` | unset | Label Studio base URL |
  | `LABEL_STUDIO_API_KEY` | unset | Label Studio access token |

  `LABEL_STUDIO_HOSTNAME` is accepted as an alias of `LABEL_STUDIO_URL` for
  continuity with the old backend.
- The previous README said the default epochs were 40 while the code used 100; 100
  is what actually ran and is the default here.
- Data directory layout, owned by the filesystem adapters: `examples/images`,
  `examples/labels`, `examples/classes.txt`, `runs/<run name>/weights/best.pt`,
  `runs/<run name>.json` (the run record), `dataset/` (rebuilt on every Training
  Run), `cache/` (downloaded Task images), `hard_frames/`.
- Effective settings are logged once at startup with the API key redacted.

### Media resolution (Label Studio adapter)

- An absolute http(s) URL is fetched directly; the access token header is added when
  the URL's host matches the Label Studio URL.
- A reference beginning with `/data/` (uploads and local-files) is prefixed with the
  Label Studio hostname from setup, else from settings, and fetched with the token.
  Setup's credentials win as a pair: the configured token is sent to a hostname setup
  named only when it is the configured host. With no hostname available the
  resolution fails with a reason.
- A reference that is an existing local path is read directly.
- Fetched bytes are cached under the data directory keyed by a hash of the resolved
  URL, so the same Label Studio path on a different hostname is fetched again; then
  decoded with OpenCV. Decoding failure is `MediaUnavailable`.

### Detector lifecycle (YOLO adapter and detector cache)

- The detector cache holds the current Detector and the Checkpoint it came from.
  Before each PredictTasks it compares the repository's latest Checkpoint with the
  loaded one and reloads under a lock when the latest is newer or a force reload
  was requested. Requests already inside detect() finish on the old Detector.
- The YOLO adapter loads with ultralytics, predicts with the configured confidence
  floor and image size, and emits Detections with boxes normalized by the original
  image size and labels taken from the Checkpoint's class-name map.
- With no Checkpoint anywhere, the NullDetector serves; health reports version
  `none` and a startup log line says so plainly.

### Example Store (filesystem adapter, ADR 0003)

- Save writes the image as JPEG named by task id, a text file of `class cx cy w h`
  lines, and maintains `classes.txt` (label names in first-seen order; a new label
  is appended). Saving the same task id overwrites both files.
- positive_count counts label files with at least one line. The existing 333
  Examples from the previous backend are readable as-is once copied.

### Training (subprocess adapter, ADR 0002)

- start() refuses if a run is active, otherwise spawns the package's `train`
  command as a detached process with the same interpreter and inherited
  environment, and writes the run record.
- active() and refresh() derive status from the record: pid liveness via psutil,
  success via a Checkpoint in the run's output newer than the start time.
- The in-process `train` command: collects positive Examples; seeded shuffle;
  15% validation with a minimum of one; copies into a fresh dataset folder; writes
  the data description with class names from `classes.txt`; runs ultralytics
  training with the configured base model, epochs, patience, image size, batch,
  device and optional learning rate, zero data-loader workers, and the run name;
  exits non-zero if fewer than the minimum Examples exist.

### Mining (OpenCV adapter and CLI)

- The FrameSource wraps OpenCV video capture. The CLI prints progress every 200
  sampled frames and a final summary line, and writes the picks.

### Logging and errors

- Standard-library logging, one logger per adapter and one for the application.
- Domain and application errors are typed: `InvalidLabelConfig`,
  `MediaUnavailable`, `TrainingAlreadyActive`, `NotEnoughExamples`,
  `MissingLabelStudioCredentials`. The HTTP adapter maps `InvalidLabelConfig` to
  400; prediction failures are isolated per Task; webhook work runs in the
  background and its outcome, success or failure, is logged, never returned.

### Migration from the previous backend

- One documented, one-time copy: the old `ls_data/images` and `ls_data/labels`
  into `examples/images` and `examples/labels` under the new data directory; a
  `classes.txt` containing `pipe`; the old `runs/pipe_ls/weights/best.pt` into
  `runs/active/weights/best.pt` (or set `LB_CHECKPOINT` to it in place).
- The README carries a table mapping the old `PIPE_*` names to the new `LB_*`
  names.
- The Label Studio project needs no change; the same Connect Model URL works.

### Documentation

- README: what it is, connect steps, run commands, settings table, migration,
  Docker `host.docker.internal` note, how to add a Detector.
- CONTEXT.md and ADRs 0001 to 0003 already exist in this repo and are the source of
  vocabulary and rationale.

## Testing Decisions

### What makes a good test here

A good test drives the system through an inbound adapter or a port and asserts on
responses and on state visible through ports. It never asserts on private methods,
log text, or the order of internal calls. Fakes are small hand-written in-memory
implementations of the ports, not mocking-library objects. Every test must run with
no GPU, no torch import, no network and no Label Studio, except the one explicitly
gated GPU test.

### Seams

- Primary seam, and the only one for behaviour: the composition root takes port
  implementations. Tests build the FastAPI app with a scripted DetectorFactory, an
  in-memory CheckpointRepository, an in-memory ExampleStore, a FakeTrainer that
  records starts and lets a test flip a run's status, a FakeMediaResolver that
  returns a fixed image per reference or raises, a scripted
  LabelStudioProjectClient, and a synchronous BackgroundRunner. They drive the app
  with FastAPI's test client and assert on responses and fake state. This one seam
  covers every use case, the HTTP contract, the retrain rule end to end, label
  mapping, per-Task failure isolation, force reload, hot reload and the skipped
  paths.
- Domain unit tests for the pure rules: coordinate conversions (round trips between
  normalized, Label Studio percent and YOLO forms), label-config parsing including
  each rejection, the retrain rule, Hard Frame scoring and selection, label mapping,
  and Training Run status derivation.
- One contract test per real adapter, each against its port:
  - filesystem ExampleStore in a temporary directory: save, replace, positive
    count, classes file growth, exact YOLO line format, and reading a copy of the
    legacy layout;
  - Label Studio client and media resolver against an httpx mock transport: token
    header, each reference form, missing hostname, HTTP and decode failures,
    caching;
  - subprocess Trainer with a stub `train` command that writes a fake Checkpoint or
    exits without one: running to succeeded, running to failed, refusal of a second
    start, recovery from a dead pid;
  - OpenCV FrameSource against a tiny video generated in the test;
  - YOLO Detector marked `gpu`: runs only when `LB_TEST_CHECKPOINT` points at a
    Checkpoint, and asserts real Detections on a fixture image, otherwise skipped.
- `mine` is tested through its use case with the fakes and a temporary output
  folder.
- An architecture test runs the import-linter contracts.

### Prior art

None in this repo; it is greenfield. The nearest established pattern is FastAPI's
test client combined with a composition root that accepts overrides.

## Out of Scope

- The classical colour-and-watershed detector and any dependency on the
  pipe-counting package. The Detector port is where it would return.
- Region types other than bounding boxes: segmentation, keypoints, classification.
- More than one Label Studio project or label config per server process.
- Authentication on the backend itself, Redis or any job queue, s3 or gs media
  references, a Docker image, a web UI.
- Changing the training recipe, progress reporting beyond `is_training`, metrics.
- The counting service and everything else in the pipe-counting repo.
- Migrating git history from the old folder.

## Further Notes

- Decisions the grilling deferred and this spec resolves: environment prefix is
  `LB_`; one project per process; bounding boxes only; `.env` loses to the shell.
  Each is cheap to reverse and none was worth an ADR.
- This spec is filed as GitHub issue #1 on `pergazuz/LB-anythings`
  (https://github.com/pergazuz/LB-anythings/issues/1), labelled `ready-for-agent`. This file is the in-repo copy; the issue is the tracked one.
- The spec keeps the previous backend's exact Label Studio behaviour wherever the
  old behaviour was intentional, and changes it only where the old behaviour was a
  bug: untruthful `is_training`, overlapping Training Runs, whole-batch failure on
  one bad image, and cancelled Annotations stored on the webhook path.
