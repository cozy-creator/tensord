# GPU launch policy ownership

- Owner: machine_front_door; parent root owns GPU/rental/actual UID hardware gates.
- Purpose: optional root-sealed executor UID/GID through existing SDK trampoline, exact
  peer credentials and necessary owned-path permissions for matched old/new comparison.
- Worktree: /home/fidika/cozy/.worktrees/cozy-machine/4-gpu-launch-policy-20261002.
- Branch: feat/4-gpu-launch-policy-20261002.
- Base: freshly fetched origin/master 9ef3799b6a96e719f29f6a94ff9e9eff093153fe,
  fast-forwarded GPU-service prerequisite 5373ffcdabbcfb9296e814d1c9d8b23d46374b23.
- All code/build/tests CPU-only, no default-state/config/machine changes. This is an
  audit-critical comparison safeguard exception to the held expansion, not an additional
  public API, planner, containment framework or version floor.
