# Direct player endpoints

The persistent machine selects one media port itself, records it under its owned state,
and reuses it after restart. Its authenticated Status reports the listener port, direct
addresses and its own DTLS fingerprint. The media listener binds independently of the
loopback personal control API so LAN/Tailscale viewers can reach it.

Provider endpoints use configured public mappings. Vast's PUBLIC_IPADDR/VAST_TCP_PORT_X
mapping is documented; RunPod's generic port mapping needs a live provider check. An explicit
COZY_WEBRTC_PUBLIC_ADDRESS may configure a mapped endpoint. Missing NAT reachability is a
clear playback error; no relay is introduced.

The CLI will read that Status directly with the locally held rental/machine owner key,
identify a v1 run by request id, and build the scoped fragment link without a Hub lookup.
G's real Pion/player checks will be ported selectively, preserving deleted GC code.

Required proof: stable restart port, actual LAN/direct Status metadata, ordinary local/rental
CLI link and Pion/browser bytes/reconnect. Unit port/config checks are not browser or inference
qualification.
