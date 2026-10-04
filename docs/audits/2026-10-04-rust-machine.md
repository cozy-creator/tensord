# Rust machine independent audit

Status: in progress. Owner: Codex, 2026-10-04.

This audit evaluates the Rust machine against two requirements: correct inference under
constrained resources, and faster completion than ComfyUI on matched workloads across GPUs.
The inherited implementation and handoff are evidence, not a binding architecture.

## Baselines

| Repository | Fetched origin/master |
|---|---|
| cozy-machine | cf902bf86d9f980af2e8c912c3d76fef1f5c53b0 |
| cozy-runtime | b2a9373182db0536d177721b74a8fcacf705d80f |
| tensorfs | dd203e1a88329f30bbc507de28098a8441d082ab |
| cozy CLI | dc7c3ecfdbd5d49d00d0ad700b1292e201a98d4d |
| tensorhub | 42b17fe91a3c1db3d43c1b6d876fdb159491ab9a |

The machine actually links TensorFS d1aad922c2586574754b6c5cea764859aa6f0246,
which is older than TensorFS master. Runtime's machine pin is inspected separately.
Each repository has an owned audit worktree branched from the fetched remote master.
Primary checkouts and other worktrees are preserved.

## Evidence standard

- CONFIRMED SOURCE: a concrete failure path follows from the pinned code.
- CONFIRMED CPU: a reproduction or relevant check ran on this audit's snapshot.
- CONFIRMED GPU: actual inference or device execution ran on this audit's snapshot.
- REPORTED: inherited measurement; its inputs, revision and limits must remain attached.
- PLAUSIBLE: a hypothesis requiring additional evidence.

Historical tests, source review and component pilots are not current end-to-end qualification.

## Coverage

The review covers ownership and process topology; scheduling and concurrency; VRAM, host
memory and disk accounting; paging and OOM recovery; multi-GPU collectives; process fencing,
cancellation and restart; journal acceptance and output custody; GC roots and retention;
authentication and package boundaries; CLI and Hub consumers; API evolution; migration and
release gates; benchmark correctness; redundant implementations and simplification.

Findings, validation and design recommendations follow after inspection.
