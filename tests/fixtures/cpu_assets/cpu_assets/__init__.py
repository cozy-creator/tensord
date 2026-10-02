"""Real inference with scalar/list files and opaque user metadata."""
import msgspec
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from cozy_runtime.author import App, Context, Outputs, Telemetry, FileAsset

app = App()

class Request(msgspec.Struct):
    samples: list[list[float]]
    iterations: int = 2
    seed: int = 19

class Response(msgspec.Struct):
    predictions: list[int]
    report: FileAsset
    reports: list[FileAsset]
    opaque: dict[str, str]

@app.entrypoint
def classify(payload: Request, ctx: Context, out: Outputs, tel: Telemetry) -> Response:
    features, labels = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=500, random_state=payload.seed).fit(features, labels)
    predictions = model.predict(payload.samples).tolist()
    report = out.save_bytes(msgspec.json.encode({"predictions": predictions}), media_type="application/json")
    reports = [out.save_bytes(msgspec.json.encode({"predictions": predictions, "index": index}),
                              media_type="application/json") for index in range(2)]
    return Response(predictions, report, reports,
                    {"asset_ref": "authored user metadata", "digest": "unchanged"})
