"""Actual CPU classifier inference through Runtime's typed author API."""
from __future__ import annotations

import msgspec
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from cozy_runtime.author import App, Context, FileAsset, Outputs, Telemetry

app = App()


class Request(msgspec.Struct):
    samples: list[list[float]]
    iterations: int = 1


class Response(msgspec.Struct):
    predictions: list[int]
    probabilities: list[list[float]]
    iterations: int
    report: FileAsset


@app.entrypoint
def classify(payload: Request, ctx: Context, out: Outputs, tel: Telemetry) -> Response:
    features, labels = load_iris(return_X_y=True)
    classifier = LogisticRegression(max_iter=500, random_state=19).fit(features, labels)
    samples = np.asarray(payload.samples, dtype=np.float64)
    on_step = tel.step_callback(payload.iterations, stage="classifier inference")
    with tel.stage("classifier inference"):
        for index in range(payload.iterations):
            ctx.raise_if_cancelled()
            predictions = classifier.predict(samples).tolist()
            probabilities = classifier.predict_proba(samples).tolist()
            on_step(index)
    body = {"predictions": predictions, "probabilities": probabilities,
            "iterations": payload.iterations}
    report = out.save_bytes(msgspec.json.encode(body), media_type="application/json")
    return Response(predictions, probabilities, payload.iterations, report)
