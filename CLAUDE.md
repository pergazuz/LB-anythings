# CLAUDE.md

Guidance for agents working in this repo. Vocabulary lives in `CONTEXT.md`,
decisions in `docs/adr/`, and the current spec in `docs/specs/`.

## Agent skills

### Issue tracker

Issues and specs are GitHub Issues on `pergazuz/LB-anythings`, driven with the
`gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels, each string equal to its role name: `needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root.
See `docs/agents/domain.md`.

## Working on it

### The checks

All five, before every commit. CI runs the same ones and fails the build on any of them:

```powershell
uv sync --no-default-groups --group dev   # the tests need neither the ML stack nor MLflow
uv run pytest -q
uv run mypy                               # strict, over src and tests
uv run ruff check .
uv run ruff format .                      # the one that gets forgotten; CI checks it
uv run lint-imports                       # the dependency rule, enforced
```

`uv` runs everything: there is no bare `python`. A plain `uv sync` also installs the ML group
(torch cu128, ultralytics, opencv, ~2.6 GB) and MLflow -- both optional, and both absent in CI.
So `torch`, `ultralytics`, `cv2` and `mlflow` are imported **inside the function that uses
them**, never at module scope, even in the adapters that own them. That is what lets the whole
package be imported, checked and tested without them, and it is worth a comment at the import
saying which group it belongs to.

### Where a change goes

The code is a hexagon and dependencies only ever point inward:

- **`domain/`** -- rules that are true regardless of how anything is stored or served. Pure
  Python, no third-party imports at all. A rule with a decision in it belongs here.
- **`application/`** -- use cases, and in `ports.py` the `Protocol` each one needs from the
  world. No frameworks, no drivers, no `httpx`, no `subprocess`.
- **`adapters/`** -- everything that knows a library, a wire format or an operating system.
  Inbound (`http`, `cli`) and outbound (one package per technology). **Adapters may not import
  one another**; the composition root wires them.
- **`bootstrap/`** -- settings, and the composition root that decides which adapter is which.

Four import-linter contracts hold that shape, and they run as a pytest test as well as in CI.
Adding an outbound adapter means adding it to the independence contract in `pyproject.toml`.

### Tests

Test-first, at the seam the change is about: the FastAPI `TestClient` for HTTP behaviour, the
use case for orchestration, the pure function for a rule. Ports are faked by hand in
`tests/fakes.py` -- there is no mocking library, and adding one would be a change worth an ADR.
HTTP adapters are tested against `httpx.MockTransport`; process adapters spawn real processes.

Name a test as a sentence about behaviour, and give it a docstring when the reason it matters
is not obvious from the name. Tests are documentation that fails.

### Verify against the real thing

Stand-ins keep hiding real defects: the Label Studio auth scheme, a token Label Studio issues
and then rejects, `START_TRAINING` on every submit, a CI job that was not testing what it
claimed. Before closing a ticket that touches the Label Studio protocol, paths, training or the
Stack, run it against a real Label Studio and say in the ticket what you saw.

### Vocabulary

`CONTEXT.md` is binding, including its `_Avoid_` lines: use those words in code, tests, docs,
commit messages and ticket comments. A new concept gets an entry there, in the same shape.

### Decisions

A choice a future reader would otherwise re-litigate -- or one taken against the obvious
alternative -- goes in `docs/adr/NNNN-<what-was-decided>.md`, written as the decision, its
context and its consequences. Cite sources where the reasoning rests on them.

### Commits

Say what changed and why it changed, in the vocabulary above; the diff already says how.
Reference the ticket. **No `Co-Authored-By` and no "Generated with" trailers**, ever.
