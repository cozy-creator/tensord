"""Read only admitted native source metadata; no model construction or Store open."""

import hashlib
import json
import math

import msgspec
from cozy_runtime.author import App, Context, invocable
from cozy_runtime.derive.quantization import QuantizationSource


class ConfigRow(msgspec.Struct, frozen=True):
    name: str
    json: str
    sha256: str


class ComponentRow(msgspec.Struct, frozen=True):
    name: str
    tensors: int


class Metadata(msgspec.Struct, frozen=True):
    manifest: str
    manifest_length: int
    configs: list[ConfigRow]
    components: list[ComponentRow]
    geometry_json: str


class PartRow(msgspec.Struct, frozen=True):
    role: str
    dtype: str
    shape: list[int]
    bytes: int
    sha256: str


class TensorRow(msgspec.Struct, frozen=True):
    name: str
    dtype: str
    shape: list[int]
    encoding: str
    parts: list[PartRow]
    logical_value_sha256: str | None


class Inventory(msgspec.Struct, frozen=True):
    manifest: str
    manifest_length: int
    tensors: list[TensorRow]


_WIDTHS = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f8_e4m3fn": 1,
    "f8_e5m2": 1,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "u8": 1,
    "bool": 1,
}
_PLAIN = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"


def part_digest(ctx, capability, component, key, role, part):
    """Hash only the admitted part with a reusable one MiB buffer."""
    length = math.prod(part.shape) * _WIDTHS[part.dtype]
    buffer = bytearray(min(length, 1 << 20))
    h = hashlib.sha256()
    for offset in range(0, length, len(buffer)):
        ctx.raise_if_cancelled()
        chunk = memoryview(buffer)[: min(len(buffer), length - offset)]
        capability.read_part_into(component, key, role, offset, chunk)
        h.update(chunk)
    return PartRow(role, part.dtype, list(part.shape), length, h.hexdigest())


@invocable
async def inventory(ctx: Context, *, source: QuantizationSource) -> Inventory:
    with ctx.tensorfs_source(source) as capability:
        actual = capability.inspect()
        rows = []
        for component, tensors in sorted(actual.components.items()):
            for key, tensor in sorted(tensors.items()):
                parts = [
                    part_digest(ctx, capability, component, key, role, part)
                    for role, part in sorted(tensor.parts.items())
                ]
                plain = (
                    tensor.encoding == _PLAIN
                    and len(parts) == 1
                    and parts[0].role == "value"
                    and parts[0].dtype == tensor.logical_dtype
                    and tuple(parts[0].shape) == tuple(tensor.shape)
                )
                rows.append(
                    TensorRow(
                        component + "/" + key,
                        tensor.logical_dtype,
                        list(tensor.shape),
                        tensor.encoding,
                        parts,
                        parts[0].sha256 if plain else None,
                    )
                )
        capability.check()
        return Inventory(actual.source.manifest, actual.source.length, rows)


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
            [
                ConfigRow(name, data.decode("utf-8"), hashlib.sha256(data).hexdigest())
                for name, data in actual.configs.items()
            ],
            [ComponentRow(name, len(rows)) for name, rows in actual.components.items()],
            json.dumps(rows, sort_keys=True, separators=(",", ":")),
        )


app = App()
app.job(metadata, accelerator=False)
app.job(inventory, accelerator=False)
