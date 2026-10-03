#!/bin/sh
# Development CUDA image only (same-pod A/B gate): run the machine arm named in
# /var/lib/cozy/machine-arm (rust, the default, or go). When the arm exits and
# /var/lib/cozy/machine-arm-switch exists, remove it and start the named arm; any other exit
# (a stop, an accepted idle release) ends the container with the arm's status.
while :; do
  case $(cat /var/lib/cozy/machine-arm 2>/dev/null || echo rust) in
    go) arm=/opt/cozy/python/bin/cozy-machine ;;
    *) arm=/opt/cozy/machine/cozy-machine ;;
  esac
  "$arm" & child=$!
  trap 'kill -TERM $child 2>/dev/null' TERM INT
  status=0
  while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
  trap - TERM INT
  [ -e /var/lib/cozy/machine-arm-switch ] || exit "$status"
  rm -f /var/lib/cozy/machine-arm-switch
done
