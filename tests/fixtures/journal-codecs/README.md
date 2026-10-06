These four binary fixtures were emitted by the native machine's pre-retirement worker-data
writer, from source preserved in the task evidence (`machine-domain/old-codec-fixtures.rs`).
They cover model choices, product/source custody, intake receipts and terminal event pages.

The private archive codec reads and writes these same stored bytes after worker-protocol
generation is removed. This protects existing local storage; it is not a peer version or
serialized-message equality gate.
