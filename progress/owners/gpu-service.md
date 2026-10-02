# GPU public-service integration ownership

- Owner: machine_front_door, coordinated by root.
- Purpose: authenticated durable GPU application execution through the public
  machine API, shared engine/journal/custody and existing DeviceExecutor/model broker.
- Branch: feat/4-gpu-service-20261002.
- Worktree: /home/fidika/cozy/.worktrees/cozy-machine/4-gpu-service-20261002.
- Base: fetched origin/master 9ef3799b6a96e719f29f6a94ff9e9eff093153fe.
- Integrated prerequisite: root PR #12 branch feat/3-service-integration-20261002
  through 3a5ec84488c4cd5e6e91929d6fc0c84f45fc3b4f, fast-forwarded into this branch.
- Development and tests are CPU-only, heavy work nice19/max2 jobs under heavy.lock.
  Root owns GPU/NVML, hardware qualification and rentals. No default daemon,
  existing machine, primary checkout or unrelated work is modified.
- Ordinary CLI GPU inference and memory gates remain required; no complete Degree1
  or full machine-stack replacement claim follows from source/CPU tests alone.
