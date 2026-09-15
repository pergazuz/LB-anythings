# LB-anythings

A Label Studio ML backend that proposes bounding boxes for images, learns from the
human corrections, and retrains itself as annotations accumulate. It is detector-
agnostic; the first Detector it serves is the pipe detector from the pipe-counting
project.

## Language

### People

**Annotator**:
The person correcting Predictions inside Label Studio.
_Avoid_: labeler, user

**Operator**:
The person who runs the backend, connects it to a project, and decides when to train.
_Avoid_: admin, developer (a developer changes the code; an operator runs it)

### Labelling

**Task**:
One Label Studio task: one image to be annotated.
_Avoid_: item, sample, image (the image is the Task's content, not the Task)

**Annotation Target**:
What the label config tells us to address: the control tag, the image tag it points
at, the task field holding the image, and the label values that exist.
_Avoid_: label config (the raw XML the Annotation Target is read from), schema

**Prediction**:
The regions a Detector proposes for one Task, sent to Label Studio for an Annotator
to correct.
_Avoid_: pre-annotation (user-facing docs only), result, inference

**Annotation**:
The human-corrected regions for one Task, received from Label Studio.
_Avoid_: completion, label, correction

### Detecting

**Detector**:
The thing that turns an image into Detections. A loaded YOLO Checkpoint is one
Detector.
_Avoid_: model, backend, predictor

**Detection**:
One box with a confidence score and a label, produced by a Detector.
_Avoid_: det, result, box (a box is geometry; a Detection is the box plus what the
Detector says about it)

**Checkpoint**:
A trained weights file a Detector is loaded from. Its name is the version Label
Studio sees.
_Avoid_: weights, model, .pt, model version

### Learning

**Example**:
One image together with its ground-truth boxes, kept for training. There is one per
Task; a later Annotation of the same Task replaces it.
_Avoid_: sample, label, training data

**Ground-Truth Box**:
A box an Annotator confirmed, with the label they gave it. What an Example is made of.
_Avoid_: region (Label Studio's word for it on the wire), detection (a Detector's guess)

**Training Set**:
All Examples collected so far. The train/validation split is not a domain concept.
_Avoid_: dataset, ls_data, corpus

**Training Run**:
One execution that consumes the Training Set and produces a Checkpoint. At most one
is active at a time.
_Avoid_: fit, job, training session

**Retrain Threshold**:
The fewest new Examples worth a Training Run. It is a floor, not the whole rule: the
Training Set must grow since the last run by the threshold *or* by a share of its own
size, whichever is larger, so retraining is frequent while the set is small and rare
once it is large (ADR 0004).
_Avoid_: train_every, batch size, interval

**Run Trigger**:
What launched a Training Run: the Retrain Threshold, the Start Training button, or the
command line. Known only to whatever launched it, so it is recorded at launch or not at
all.
_Avoid_: source, cause, reason (a reason is why a run did *not* start)

**Hard Frame**:
A video frame the Detector is uncertain about, and therefore worth labelling next.
_Avoid_: hard example (an Example is already labelled; a Hard Frame is not), hard
case, difficult frame
