# Cozy machine

Keep implementation in this repository; the existing Runtime, TensorFS, Creator and Hub
checkouts are concurrently owned. Use released TensorFS dependencies, not local path overrides
or copied storage/model implementations.

Primary checkout is a mirror of origin/master. Work in a named owned worktree; preserve all
unrelated work. Use Paul Fidika <paul@fidika.com> as author and committer, without agent trailers.
Use uv for Python. Use CodeGraph first if an index exists; do not create one implicitly.

The CPU service never loads CUDA/NVML. GPU work or rentals require explicit session authority.
Records at boundaries are typed; peer versions never gate baseline operations. No time-based
process kills, hidden environment behavior switches, smaller substitute requests or mock
inference qualification.

Qualify vertical slices with actual process/socket/storage/inference paths and record their
limits. This repository is experimental until ordinary Cozy CLI, consumer and release gates pass.
