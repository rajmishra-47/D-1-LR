from __future__ import annotations

from typing import Any, Mapping

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

INTEGER_COLUMNS = {
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "Age",
}

FLOAT_COLUMNS = {
    "BMI",
    "DiabetesPedigreeFunction",
}


def preprocess_input(payload: Mapping[str, Any]) -> list[float]:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in payload]
    extra_columns = [column for column in payload if column not in EXPECTED_COLUMNS]

    if missing_columns:
        raise ValueError(f"Missing required fields: {', '.join(missing_columns)}")
    if extra_columns:
        raise ValueError(f"Unexpected fields: {', '.join(extra_columns)}")

    processed_values: list[float] = []
    for column in EXPECTED_COLUMNS:
        value = payload[column]
        if column in INTEGER_COLUMNS:
            processed_values.append(int(value))
        elif column in FLOAT_COLUMNS:
            processed_values.append(float(value))
        else:
            raise ValueError(f"Unsupported column: {column}")

    return processed_values
