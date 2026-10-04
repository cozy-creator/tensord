#!/bin/sh
# Development worker image with the Rust machine: one layer appended to the current qualified
# tensorhub/worker image of KIND (cpu | cuda). `push` pushes it untagged by digest, never onto a
# tag (record the digest in D2/IMAGES.md); without it the staged layer is left for a local boot.
# A third argument names a directory of Runtime/TensorFS wheels the Rust machine's executors use.
# Rent it with `cozy rental new <sku> --image=sha256:<digest>`. The cuda image also keeps the
# Go agent behind a dev dispatcher (/var/lib/cozy/machine-arm = rust | go) for same-pod A/B.
set -eu
kind=$1; push=${2:-}; own_wheels=${3:-}; case $kind in cpu) tag=cpu-linux-x86 ;; cuda) tag=torch2.14.0-cu130-linux-x86 ;; *) echo "usage: $0 cpu|cuda" >&2; exit 2 ;; esac
repo=$(cd "$(dirname "$0")/.." && pwd); commit=$(git -C "$repo" rev-parse HEAD); base=$(crane digest "tensorhub/worker:$tag")
target=${CARGO_TARGET_DIR:-$repo/target}/bookworm; stage=$(mktemp -d); mkdir -p "$target" "$stage/opt/cozy/machine" "$stage/usr/local/bin" "$stage/etc/cozy"

# The base is glibc 2.36 (bookworm); a host build needs newer symbols.
nice -n 19 docker run --rm --runtime=runc --user "$(id -u):$(id -g)" -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 -v "$HOME/.cargo:/cargo" -v "$target:/target" -v "$repo:/src" -w /src \
  rust:1.91-bookworm cargo build --release --locked --offline -j 2 --bin cozy-machine
install -m 0755 "$target/release/cozy-machine" "$stage/opt/cozy/machine/cozy-machine"

# Optional: this machine's own executor SDK (Runtime/TensorFS wheels), apart from the Go agent's pair.
if [ -n "$own_wheels" ]; then mkdir -p "$stage/opt/cozy/machine/wheels" && cp "$own_wheels"/*.whl "$stage/opt/cozy/machine/wheels/"; fi
if [ "$kind" = cuda ]; then
  install -m 0755 "$repo/scripts/machine-arm-dispatcher.sh" "$stage/usr/local/bin/cozy-machine"
  echo '{"startup_update":"off","agent":"explicit"}' > "$stage/etc/cozy/software-policy.json"
else
  ln -s /opt/cozy/machine/cozy-machine "$stage/usr/local/bin/cozy-machine"; rmdir "$stage/etc/cozy" "$stage/etc"
fi
find "$stage" -mindepth 1 -type d -exec chmod 0755 {} +  # a layer's directory entries replace the base's modes
tar --numeric-owner --owner=0 --group=0 --mtime=@0 --sort=name -C "$stage" -czf "$stage.tgz" $(ls -A "$stage")
[ "$push" = push ] || { echo "staged $kind layer: $stage (base tensorhub/worker@$base)"; exit 0; }
pushed=$(crane mutate "tensorhub/worker@$base" --append "$stage.tgz" -l cozy.machine_impl=rust -l "cozy.machine_impl.commit=$commit" -l "cozy.machine_impl.base=$base")
digest=${pushed#*@}; [ "$digest" != "$pushed" ] || { echo "push failed" >&2; exit 1; }
echo "$kind $digest commit=$commit base=$base"
