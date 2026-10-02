"""Real CPU classification with a deferred WebP artifact through the normal SDK."""
import msgspec
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from cozy_runtime.author import App, Context, ImageAsset, ImageFrame, Outputs, Telemetry

app = App()


class Request(msgspec.Struct):
    samples: list[list[float]]


class Response(msgspec.Struct):
    predictions: list[int]
    image: ImageAsset


@app.entrypoint
def render(payload: Request, ctx: Context, out: Outputs, tel: Telemetry) -> Response:
    features, labels = load_iris(return_X_y=True)
    classifier = LogisticRegression(max_iter=500, random_state=19).fit(features, labels)
    ctx.raise_if_cancelled()
    predictions = classifier.predict(np.asarray(payload.samples, dtype=np.float64)).tolist()
    colors = [(230, 50, 50), (50, 230, 50), (50, 50, 230)]
    pixels = bytes(channel for y in range(32) for x in range(32)
                   for channel in colors[predictions[min(x * len(predictions) // 32,
                                                         len(predictions) - 1)]])
    image = out.save_image(ImageFrame(32, 32, pixels), format="webp")
    return Response(predictions, image)
