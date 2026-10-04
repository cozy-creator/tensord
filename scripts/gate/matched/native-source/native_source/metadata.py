# /// script
# dependencies = ["cozy-runtime", "msgspec"]
# ///
"""Read only admitted native source metadata; no model construction or Store open."""

import hashlib
import json

import msgspec
from cozy_runtime.author import App, Context, invocable
from cozy_runtime.derive.quantization import QuantizationSource


class Metadata(msgspec.Struct, frozen=True):
    manifest: str
    manifest_length: int
    configs_json: dict[str, str]
    configs_sha256: dict[str, str]
    component_tensors: dict[str, int]
    geometry_json: str


@invocable
async def metadata(ctx: Context, *, source: QuantizationSource) -> Metadata:
    ctx.raise_if_cancelled()
    with ctx.tensorfs_source(source) as capability:
        actual = capability.inspect()
        rows = []
        for component, tensors in actual.components.items():
            for key, tensor in tensors.items():
                rows.append(
                    {
                        "component": component,
                        "key": key,
                        "logical_dtype": tensor.logical_dtype,
                        "shape": tensor.shape,
                        "encoding": tensor.encoding,
                        "parts": {
                            role: {"dtype": part.dtype, "shape": part.shape}
                            for role, part in tensor.parts.items()
                        },
                    }
                )
        return Metadata(
            actual.source.manifest,
            actual.source.length,
            {name: data.decode("utf-8") for name, data in actual.configs.items()},
            {
                name: hashlib.sha256(data).hexdigest()
                for name, data in actual.configs.items()
            },
            {name: len(rows) for name, rows in actual.components.items()},
            json.dumps(rows, sort_keys=True, separators=(",", ":")),
        )


app = App()
app.job(metadata, accelerator=False)
