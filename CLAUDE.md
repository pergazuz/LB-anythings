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
