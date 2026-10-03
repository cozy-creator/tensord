#!/bin/sh
# Development worker image with the Rust machine: one layer appended to the current qualified
# tensorhub/worker image of KIND (cpu | cuda). `push` pushes it untagged by digest, never onto a
# tag (record the digest in D2/IMAGES.md); without it the staged layer is left for a local boot.
# Rent it with `cozy rental new <sku> --image=sha256:<digest>`. The cuda image also keeps the
# Go agent behind a dev dispatcher (/var/lib/cozy/machine-arm = rust | go) for same-pod A/B.
set -eu
kind=$1; push=${2:-}; case $kind in cpu) tag=cpu-linux-x86 ;; cuda) tag=torch2.14.0-cu130-linux-x86 ;; *) echo "usage: $0 cpu|cuda" >&2; exit 2 ;; esac
repo=$(cd "$(dirname "$0")/.." && pwd); commit=$(git -C "$repo" rev-parse HEAD); base=$(crane digest "tensorhub/worker:$tag")
target=$HOME/cozy/.cargo-target/cozy-machine-bookworm; stage=$(mktemp -d); mkdir -p "$target" "$stage/opt/cozy/machine" "$stage/usr/local/bin" "$stage/etc/cozy"

# The base is glibc 2.36 (bookworm); a host build needs newer symbols.
nice -n 19 docker run --rm --user "$(id -u):$(id -g)" -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target -v "$HOME/.cargo:/cargo" -v "$target:/target" -v "$repo:/src" -w /src \
  rust:1.91-bookworm cargo build --release --locked --offline -j 2 --bin cozy-machine
install -m 0755 "$target/release/cozy-machine" "$stage/opt/cozy/machine/cozy-machine"

# The installer helper: its own environment over the image's interpreter, holding the client wheel.
# Both kinds share python:3.12.12-slim-bookworm at /opt/cozy/python, so the cpu image builds it.
uv build -q --wheel --out-dir "$stage/opt/cozy/machine" "$repo"
docker run --rm --entrypoint sh -v "$stage/opt/cozy/machine:/opt/cozy/machine" "tensorhub/worker@$(crane digest tensorhub/worker:cpu-linux-x86)" -c \
  'uv venv -q --python /opt/cozy/python/bin/python3 /opt/cozy/machine/helper && uv pip install -q --python /opt/cozy/machine/helper/bin/python /opt/cozy/machine/*.whl"[installer]" && chown -R '"$(id -u):$(id -g)"' /opt/cozy/machine'

if [ "$kind" = cuda ]; then
  install -m 0755 "$repo/scripts/machine-arm-dispatcher.sh" "$stage/usr/local/bin/cozy-machine"
  echo '{"startup_update":"off","agent":"explicit"}' > "$stage/etc/cozy/software-policy.json"
else
  ln -s /opt/cozy/machine/cozy-machine "$stage/usr/local/bin/cozy-machine"; rmdir "$stage/etc/cozy" "$stage/etc"
fi
find "$stage" -mindepth 1 -type d -exec chmod 0755 {} +  # a layer's directory entries replace the base's modes
tar --numeric-owner --owner=0 --group=0 --mtime=@0 --sort=name -C "$stage" -czf "$stage.tgz" $(ls -A "$stage")
[ "$push" = push ] || { echo "staged $kind layer: $stage (base tensorhub/worker@$base)"; exit 0; }
digest=$(crane mutate "tensorhub/worker@$base" --append "$stage.tgz" -l cozy.machine_impl=rust -l "cozy.machine_impl.commit=$commit" -l "cozy.machine_impl.base=$base" | cut -d@ -f2)
echo "$kind $digest commit=$commit base=$base"
