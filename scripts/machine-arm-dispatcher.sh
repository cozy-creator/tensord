#!/bin/sh
# Development CUDA image only (same-pod A/B gate): run the machine arm named in
# /var/lib/cozy/machine-arm (rust, the default, or go). When the arm exits and
# /var/lib/cozy/machine-arm-switch exists, remove it and start the named arm; any other exit
# (a stop, an accepted idle release) ends the container with the arm's status. Each arm's
# output goes to /var/log/cozy-machine/<arm>.log.
mkdir -p /var/log/cozy-machine
while :; do
  case $(cat /var/lib/cozy/machine-arm 2>/dev/null || echo rust) in
    go) name=go arm=/opt/cozy/python/bin/cozy-machine ;;
    # The Go arm's released TensorFS keeps /var/lib/tensorfs; the Rust arm's store stays apart.
    *) name=rust arm="env COZY_TENSORFS_ROOT=/var/lib/cozy/rust-machine/tensorfs /opt/cozy/machine/cozy-machine" ;;
  esac
  # Each arm's output is kept on the pod for diagnosis (container output is not reachable).
  $arm >>"/var/log/cozy-machine/$name.log" 2>&1 & child=$!
  trap 'kill -TERM $child 2>/dev/null' TERM INT
  status=0
  while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
  trap - TERM INT
  [ -e /var/lib/cozy/machine-arm-switch ] || exit "$status"
  rm -f /var/lib/cozy/machine-arm-switch
done
