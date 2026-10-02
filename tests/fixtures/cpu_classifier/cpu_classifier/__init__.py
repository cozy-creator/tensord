"""Actual CPU classifier inference through Runtime's typed author API."""
from __future__ import annotations

import msgspec
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from cozy_runtime.author import App, Context, FileAsset, Outputs, Telemetry

app = App()
_calls = 0


class Request(msgspec.Struct):
    samples: list[list[float]]
    iterations: int = 1
    seed: int = 19


class Response(msgspec.Struct):
    predictions: list[int]
    probabilities: list[list[float]]
    iterations: int
    report: FileAsset
    seed: int = 19
    call_sequence: int = 0


@app.entrypoint
def classify(payload: Request, ctx: Context, out: Outputs, tel: Telemetry) -> Response:
    global _calls
    _calls += 1
    features, labels = load_iris(return_X_y=True)
    order = np.random.default_rng(payload.seed).permutation(len(features))
    classifier = LogisticRegression(max_iter=500, random_state=payload.seed).fit(
        features[order], labels[order])
    samples = np.asarray(payload.samples, dtype=np.float64)
    on_step = tel.step_callback(payload.iterations, stage="classifier inference")
    with tel.stage("classifier inference"):
        for index in range(payload.iterations):
            ctx.raise_if_cancelled()
            predictions = classifier.predict(samples).tolist()
            probabilities = classifier.predict_proba(samples).tolist()
            on_step(index)
    body = {"predictions": predictions, "probabilities": probabilities,
            "iterations": payload.iterations, "seed": payload.seed, "call_sequence": _calls}
    report = out.save_bytes(msgspec.json.encode(body), media_type="application/json")
    return Response(predictions, probabilities, payload.iterations, report, payload.seed, _calls)
