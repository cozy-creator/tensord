# Cozy machine

The Rust machine: API, execution journal, scheduler, memory policy, the linked TensorFS store
(sole writer) and executor supervision. Python cozy-runtime executors run package code. It is
meant to replace the Go agent and Python worker; it is not yet a complete replacement and is not
installed as anyone's machine.

- Architecture and module owners: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Links TensorFS by git revision (`Cargo.toml`); a release pins a TensorFS tag.
- Linux 6.5+ (`SO_PEERPIDFD`). The machine process never loads CUDA or NVML.

## Build and test

Cargo needs read access to the private TensorFS repository.

```sh
cargo build --locked
cargo clippy --all-targets -- -D warnings
cargo test --locked -- --test-threads=2
uv run --locked --extra test pytest -q
```

## Release

The version is `Cargo.toml`'s; a release is the tag `v<version>` on master. This repository
publishes nothing: cozy-runtime pins the tag in `cozy-machine.tag`, its release builds it with
`task release` (manylinux_2_28 container: glibc 2.28 floor, `version --json` names the commit)
and bundles it into the Linux x86_64 wheel as `cozy-machine`. Worker images and
`cozy machine install` take it from that wheel.

## Run

```sh
target/debug/cozy-machine version --json
target/debug/cozy-machine serve --state PATH [--generations PATH] [--host-bytes N] \
  [--cpu-parallelism N] [--gpu-config FILE] [--machine-config FILE --listen ADDR]
```

`--machine-config` with `--listen` starts the authenticated TLS/gRPC API the Cozy CLI uses.
`--gpu-config` adds the GPU pool ([GPU service](docs/GPU-SERVICE.md)).
