"""H3 long-form's shape without torch: a CPU composition job renders each segment through a
child call of its own invocable, which takes the reference image and the previous segment's
context and returns a file; after every segment the parent publishes the film so far, keeps it
in its scratch and declares it a checkpoint. A resumed attempt reads its scratch back."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    App,
    AssetBound,
    Checkpoints,
    ChildCallError,
    Context,
    FileAsset,
    ImageAsset,
    OutputError,
    Outputs,
    Scratch,
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
    #: Seconds a segment takes (a cancel or pause test holds one running).
    hold: float = 0.0
    #: The one segment that holds; -1: every segment does.
    hold_at: int = -1
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
    held = payload.hold if payload.hold_at in (-1, payload.index) else 0.0
    for _ in range(int(held * 20)):
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
    hold_at: int = -1
    fail_at: int = -1


class LongFormOutput(msgspec.Struct):
    video: Video
    segments: int
    #: Each segment's file, published (appended) as it lands.
    parts: list[Video]
    #: This run's attempts so far, counted in its scratch (a resume adds one).
    attempts: int = 1
    #: Segment checkpoints an earlier attempt had already declared.
    replayed: int = 0
    #: Each segment's file as its child returned it, published as it lands (H3 shows each
    #: reference its qwen-image-2 callee generates this way).
    rendered: list[Video] = msgspec.field(default_factory=list)


async def long_form(
    ctx: Context,
    payload: LongFormInput,
    out: Outputs,
    tel: Telemetry,
    scratch: Scratch,
    checkpoints: Checkpoints,
) -> LongFormOutput:
    prefetch(render_segment)
    state = scratch.checkpoint_dir(key="long_form")
    counted = state / "attempts"
    attempts = int(counted.read_text()) + 1 if counted.exists() else 1
    counted.write_text(str(attempts))
    film, context, replayed, parts, rendered = b"", "", 0, [], []
    on_segment = tel.step_callback(len(payload.segments), stage="segments")
    for index, prompt in enumerate(payload.segments):
        ctx.raise_if_cancelled()
        try:
            result = await render_segment(
                payload=SegmentInput(
                    index,
                    prompt,
                    payload.reference,
                    context,
                    payload.hold,
                    payload.hold_at,
                    payload.fail_at,
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
        with tel.stage("assemble"):
            film += result.video.read_bytes()
            context = result.context
            parts.append(out.save_bytes(result.video.read_bytes(), media_type="video/mp4"))
            out.publish("parts", parts[-1], label=f"Segment {index + 1}")
            out.publish("rendered", result.video, label=f"Segment {index + 1} as rendered")
            rendered.append(result.video)
            kept = state / f"film-{index}"
            kept.write_bytes(film)
            replayed += checkpoints.declare(f"film-{index}", kept).replayed
            revision = out.save_bytes(film, media_type="video/mp4")
            out.publish("video", revision, label=f"Video (segments 1-{index + 1})")
        on_segment(index)
    ctx.release_gpus()
    return LongFormOutput(
        video=out.save_bytes(film, media_type="video/mp4"),
        segments=len(payload.segments),
        parts=parts,
        attempts=attempts,
        replayed=replayed,
        rendered=rendered,
    )


app.entrypoint(internal=True)(render_segment)
app.job(long_form, emits_media=True)
