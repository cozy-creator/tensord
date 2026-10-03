"""H3 long-form's shape without torch: a CPU composition job renders each segment through a
child call of its own invocable, which takes the reference image and the previous segment's
context and returns a file; after every segment the parent publishes the film so far."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    App,
    AssetBound,
    ChildCallError,
    Context,
    FileAsset,
    ImageAsset,
    OutputError,
    Outputs,
    Telemetry,
    invocable,
    prefetch,
)

app = App()

Reference = Annotated[ImageAsset, AssetBound(max_bytes=1 << 20)]
#: One declared media type, as H3's video outputs: a client saves it under a known extension.
Video = Annotated[FileAsset, AssetBound(max_bytes=64 << 20, media_types=("video/mp4",))]


class SegmentInput(msgspec.Struct):
    index: int
    prompt: str
    reference: Reference
    context: str = ""
    #: Seconds a segment takes (a cancel test holds one running).
    hold: float = 0.0
    #: This segment fails in authored code (a failure test); -1: none does.
    fail_at: int = -1


class SegmentOutput(msgspec.Struct):
    video: Video
    context: str


@invocable
async def render_segment(ctx: Context, *, payload: SegmentInput, out: Outputs) -> SegmentOutput:
    """One segment: its bytes name the reference it saw and the context it continued."""
    ctx.raise_if_cancelled()
    if payload.index == payload.fail_at:
        raise ValueError(f"segment {payload.index} cannot be rendered")
    for _ in range(int(payload.hold * 20)):
        await asyncio.sleep(0.05)
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
    hold: float = 0.0
    fail_at: int = -1


class LongFormOutput(msgspec.Struct):
    video: Video
    segments: int


async def long_form(
    ctx: Context, payload: LongFormInput, out: Outputs, tel: Telemetry
) -> LongFormOutput:
    prefetch(render_segment)
    film, context = b"", ""
    for index, prompt in enumerate(payload.segments):
        ctx.raise_if_cancelled()
        try:
            result = await render_segment(
                payload=SegmentInput(
                    index, prompt, payload.reference, context, payload.hold, payload.fail_at
                )
            )
        except ChildCallError as failure:
            # H3's long_form: the film published so far stays; the run fails with the segment.
            ctx.raise_if_cancelled()
            raise OutputError(
                f"segment {index + 1} of {len(payload.segments)} failed ({failure.code}): "
                f"{str(failure)[:512]}",
                code="segment_failed",
            ) from failure
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
