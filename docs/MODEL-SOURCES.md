# Machine model-source broker

`src/model_sources.rs` links released TensorFS core v0.3.87. The machine's trusted
pilot configuration selects a manifest and its model components. The broker uses
the native snapshot/header parser and read planner to derive object authority;
presence in the store never authorizes another manifest or component. Declared
header assets are authorized by the selected manifest. They are not executable
package descriptions.

`ModelSources::read` accepts typed `SourceRequest { manifest, role, name, length }`
and returns `SourceGrant { sha256, length, file }`. Header and object grants are
native-verified read-only regular files; asset grants are reconstructed through
the native lease/read path and exported as sealed read-only memfds. Object length
must match the selected manifest. Unknown roles fail only this operation.
The device adapter maps these fields to `model_source_read` answers with one
SCM_RIGHTS descriptor. The Runtime hook negotiates `model_sources.descriptors/1`
before `Load.descriptor_sources = true`; legacy loads remain unchanged.

The machine owns verification records and read leases. The executor's candidate
TensorFS descriptor source needs no store/catalog/GC writes. Descriptors pin
their exact inode and remain readable after broker exit. This is a cooperative
same-user immutable-store contract, not a hostile-package sandbox.

`model-source-check STORE MANIFEST COMPONENTS OUTPUT` produces a native-derived
`closure-files.txt`, the exact header, one selected object and metadata. Components
are comma separated. Initialize an owned target with `tfs store ensure`, copy only
those native paths, then run `model-source-check --verify STORE MANIFEST COMPONENTS
OUTPUT`. This uses `Store::ensure` and one native acquisition over the selected
closure to verify/admit the transferred bytes in the target's own catalog. It
does not copy another machine's SQLite/history, invoke a subprocess per object,
change source model bytes, or invent the TensorFS shard layout.

CPU qualification passed five linked-core tests: exact selected bytes, rejection
of another manifest/component and undeclared assets, sealed declared assets,
descriptor survival after broker exit, and admission of a copied closure in a
clean independent catalog, and rejection of transferred object corruption. The authoritative SDXL snapshot
`sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642`
produced its 320,558-byte header, 2,641 tensors / 3,746 read-plan items at a 4 MiB
window, and 2,605 transfer paths for unet/text_encoder/text_encoder_2/vae. Evidence
is under `outputs/cozy-machine-continued-20261002/sdxl-source-broker-cpu-verified`.
`scripts/check-model-source-descriptor.py` verifies the selected real object with
the candidate native descriptor API, without CUDA/NVML or a consumer catalog.

This gate does not qualify SDXL inference, cold start, reuse, streamed copies,
GPU ownership or host-memory accounting. The existing native lease retains a
descriptor for every selected object and can reach the OS hard FD limit; a
bounded verified-source lifetime strategy is still needed to satisfy the user's
no-size-refusal rule generally. Receiver verification rehashes each imported
object once; its real-model cold-start cost is unmeasured. Runtime's private
pinned/staging/activation/asset buffers remain executor-owned, so descriptor
reads alone do not complete Degree 1 memory ownership.
