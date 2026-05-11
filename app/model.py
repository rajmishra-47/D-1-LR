from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "Linear-Regression-model.pkl"


class ModelNotLoadedError(RuntimeError):
    pass


with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def predict(features: list[float]) -> Any:
    if model is None:
        raise ModelNotLoadedError("The model could not be loaded.")

    input_array = np.asarray(features, dtype=float).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]
