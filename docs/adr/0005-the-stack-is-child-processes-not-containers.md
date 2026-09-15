# 5. The Stack is child processes, not containers

Date: 2026-09-15

## Status

Accepted.

## Context

The loop needs three programs running at once -- Label Studio, this backend, and the MLflow UI
that shows what the Training Runs recorded -- and a Label Studio project wired to the backend:
the model connected, and **Start model training on annotation submission** switched on. That
last one fails silently when it is off. Nothing is collected, nothing ever retrains, and
nothing says so; it has already cost this project a day.

Doing all of it by hand is three terminals and a walk through the Label Studio UI, repeated for
every new project. The question is what runs it instead.

**Docker Compose** is the obvious answer and the wrong one here. Training is the point of the
backend, and it trains on a Blackwell card through torch built for CUDA 12.8. Reaching a GPU
from a container on Windows means Docker Desktop, a WSL2 backend and the NVIDIA container
toolkit -- three more things to install and keep working, in front of a machine that already
has the card, the driver and the wheels. The Operator's data is on the host too: `data/` here,
Label Studio's own store in the user profile. A compose file would spend most of its lines
mounting the host back into the containers it just built.

**A process manager** (honcho, supervisor, a `.bat`) starts three programs and stops there. The
half that actually saves the Operator is the wiring, and no process manager knows what a Label
Studio project is.

## Decision

One command, `lb-anythings up`, which starts each Service as a child process of itself, waits
until each answers, and then wires the project through Label Studio's own REST API.

- **Adopt what is already running.** A Service that answers its health URL is left alone and
  never stopped: `up` beside a Label Studio someone else started works, and so does running it
  twice.
- **Wire after, never before.** Label Studio health-checks a model backend and calls its
  `/setup` before it will accept it, so the backend has to be answering first. That ordering
  also makes a successful `up` proof that the loop works, not just that three ports are open.
- **Find or create.** With no flags the project named by `LB_PROJECT_TITLE` is adopted if it
  exists and created if it does not, which is the same command for an existing project and a
  new one. `--project N` names one exactly; `--new` insists on a new one.
- **Report, do not overrule.** `up` switches on the training toggle, because without it nothing
  works at all. It does *not* change a model connection someone made by hand -- it says so when
  interactive preannotations are off and leaves the choice alone.
- **Stop what we started, in reverse.** The backend first, Label Studio last.

## Consequences

- Stopping a Service stops the processes it spawned. Label Studio and MLflow both fork workers,
  and a worker that outlives its parent holds the port -- which would make the next `up` adopt
  a Stack that is not there. The backend is the exception: a Training Run is its child and is
  detached on purpose (ADR 0002), because it writes a Checkpoint the next `up` will serve.
  Killing it halfway spends the GPU minutes and produces nothing, so `stop_descendants` is off
  for the backend alone.
- Services are started in their own process group, so the Ctrl+C that ends `up` does not race
  the shutdown that follows it: the parent decides what stops, and in what order.
- `up` needs a Label Studio token to wire anything. Without one it still brings the Services up
  and says what to do about the rest, because a fresh Label Studio has no user to issue a token
  until someone signs up. `LABEL_STUDIO_USERNAME` and `LABEL_STUDIO_PASSWORD` cover the
  unattended case; they go to the child in its environment, not its command line, which every
  process list can read.
- Everything specific to these three programs -- Label Studio's flags, MLflow's, our own --
  lives in `bootstrap/stack.py`. The rules about when a Service is started and how a project is
  wired are a use case over ports, and are tested with no process and no socket.
- A `docker compose` deployment is still possible for someone who wants one: `up
  --no-label-studio` adopts a Label Studio in a container, and `LB_BACKEND_URL` is how it is
  told to reach the backend at `host.docker.internal`.
