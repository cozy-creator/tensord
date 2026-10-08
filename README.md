# Cozy machine

TensorD (`cozy-machine`) is the Rust daemon that owns the machine API, execution journal,
scheduling, memory policy and executor supervision. It uses embedded TensorFS to make selected
model weights and metadata available to Runtime and to fill, retain and release shared CPU weight buffers.
Python cozy-runtime executors run package code, construct models and use their own TensorFS
weight plane for GPU transfers and mappings. TensorFS is a library on both sides, with no
separate daemon.

- Architecture and module owners: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Links TensorFS by git revision (`Cargo.toml`); a release pins a TensorFS tag.
- Linux 6.5+ (`SO_PEERPIDFD`). TensorD has no CUDA context. It can retain GPU allocation
  handles exported by executors; Runtime performs the device operations. The CPU service
  does not load NVML; a configured GPU pool samples it on its GPU sampler threads.

## Build and test

Cargo needs read access to the private TensorFS repository.

```sh
cargo build --locked
cargo clippy --all-targets -- -D warnings
cargo test --locked -- --test-threads=2
uv run --locked --extra test pytest -q
```

## Release

The version is `Cargo.toml`'s; a release is the tag `v<version>`, cut on the commit the gates ran
on. This repository publishes nothing: cozy-runtime's `cozy-machine.pin` names that commit for
candidates and the tag at publish. The Runtime build runs `task release` (manylinux_2_28
container: glibc 2.28 floor, `version --json` names the commit) and bundles the binary into its
Linux x86_64 wheel as `cozy-machine`. Worker images and `cozy machine install` take it from there.

## Run

```sh
target/debug/cozy-machine version --json
target/debug/cozy-machine serve --state PATH [--generations PATH] [--host-bytes N] \
  [--cpu-parallelism N] [--gpu-config FILE] [--machine-config FILE --listen ADDR]
```

`--machine-config` with `--listen` starts the authenticated TLS/gRPC API the Cozy CLI uses.
`--gpu-config` adds the GPU pool ([GPU service](docs/GPU-SERVICE.md)).
