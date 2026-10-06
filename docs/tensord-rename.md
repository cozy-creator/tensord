# Tensord process and artifact rename

The native server process and Rust crate become `tensord`. Runtime wheels own
`.data/scripts/tensord`; the controller installs and updates that executable.
Version JSON names the process `tensord`. The deployed API remains `cozy.machine.v1`.
Current journal/event names, actor/id reservations, engine/store paths, identity,
input/output custody and grant environment contracts retain their meaning.
The embedded `cozy_machine_client` library remains an installer/executor library,
without an old-name executable alias or worker RPC adapter.

Private initial provisioning uses declared wheels/dependencies baked into the
common PyTorch Docker image and a direct `tensord` entrypoint. Upgrades are optional.
The new native updater extracts and verifies a bundled `tensord`, then activates
or rolls it back through its stable parent while retaining journal and identity.
An older native updater does not install the renamed executable; it is not hidden
behind a wrapper. The controller must never mistake an existing native root for
an old unreadable worker root merely because the process name changed.

The independently reviewed launch-authority cut removes the old owner-auth JSON
input: `COZY_AUTHORIZED_KEYS` is the sole validated launch key list. Unknown
advisory environment inputs are reported and ignored; they never grant authority.

Prove actual executable identity, native update/rollback, authorized-key launch
and ordinary CLI execution in isolated CPU fixtures. Use a new owned Cargo target.
No personal daemon, GPU, rental, deployment or publication is part of this change.

Tracker: https://github.com/cozy-creator/tracker/issues/336.
