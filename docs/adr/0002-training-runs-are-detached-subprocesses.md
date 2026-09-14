---
status: accepted
---
# A Training Run is a detached subprocess with an on-disk run record

Training pins the GPU for minutes to hours. The alternatives were an in-process
thread (a training crash or OOM takes the server with it, and a server restart
kills the run), or a job queue such as the SDK's Redis/RQ setup (infrastructure for
a single-operator tool). We spawn our own `train` command as a detached process and
write a small run record (id, pid, start time, run name). `is_training` is derived
from that record: running while the pid is alive, succeeded once a Checkpoint newer
than the start time exists, failed if the pid is gone without one. A second run is
refused while one is running.

## Consequences

- No progress reporting beyond the three states; the trainer's own log is the
  detail.
- A machine reboot mid-run leaves a record whose pid is dead and no new Checkpoint,
  which reads as failed and unblocks the next run without manual cleanup.
- Windows-safe by construction: no fork, and the trainer runs with zero data-loader
  workers.
