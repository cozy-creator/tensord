"""H3 long-form's shape without torch: a CPU composition job renders each segment through a
child call of its own invocable, which takes the reference image and the previous segment's
context and returns a file; after every segment the parent publishes the film so far."""
from __future__ import annotations

import hashlib
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    ImageAsset,
    Outputs,
    Telemetry,
    invocable,
    prefetch,
)

app = App()

Reference = Annotated[ImageAsset, AssetBound(max_bytes=1 << 20)]


class SegmentInput(msgspec.Struct):
    index: int
    prompt: str
    reference: Reference
    context: str = ""


class SegmentOutput(msgspec.Struct):
    video: FileAsset
    context: str


@invocable
async def render_segment(ctx: Context, *, payload: SegmentInput, out: Outputs) -> SegmentOutput:
    """One segment: its bytes name the reference it saw and the context it continued."""
    ctx.raise_if_cancelled()
    seen = hashlib.sha256(payload.reference.read_bytes()).hexdigest()
    body = f"{payload.index}|{payload.prompt}|{seen}|{payload.context}\n".encode()
    return SegmentOutput(
        video=out.save_bytes(body, media_type="video/mp4"),
        context=hashlib.sha256(body).hexdigest(),
    )


class LongFormInput(msgspec.Struct):
    reference: Reference
    segments: list[str]


class LongFormOutput(msgspec.Struct):
    video: FileAsset
    segments: int


async def long_form(
    ctx: Context, payload: LongFormInput, out: Outputs, tel: Telemetry
) -> LongFormOutput:
    prefetch(render_segment)
    film, context = b"", ""
    for index, prompt in enumerate(payload.segments):
        ctx.raise_if_cancelled()
        result = await render_segment(
            payload=SegmentInput(index, prompt, payload.reference, context)
        )
        film += result.video.read_bytes()
        context = result.context
        revision = out.save_bytes(film, media_type="video/mp4")
        out.publish("video", revision, label=f"Video (segments 1-{index + 1})")
    ctx.release_gpus()
    return LongFormOutput(
        video=out.save_bytes(film, media_type="video/mp4"), segments=len(payload.segments)
    )


app.entrypoint(internal=True)(render_segment)
app.job(long_form, emits_media=True)
