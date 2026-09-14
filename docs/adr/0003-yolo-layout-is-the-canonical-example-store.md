---
status: accepted
---
# The Example Store's on-disk format is the YOLO layout

Examples could have been stored in a neutral format (for example JSON sidecars) and
exported to a training layout on each Training Run. We store them directly in the
YOLO layout: an images directory, one text file of normalized centre/width/height
boxes per image, and a classes file mapping label names to class indices. The 333
Examples already collected by the previous backend are then reusable with a copy,
the trainer reads the store without a conversion step, and no other part of the
system sees the format because it sits behind the Example Store port.

## Consequences

- A trainer for a non-YOLO framework would convert from this layout at training
  time, which is the step we skipped for YOLO.
- Label names live in the classes file; class indices in the box files are only
  meaningful together with it.
