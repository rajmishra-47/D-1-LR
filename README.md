# FastAPI Linear Regression Prediction Server

This project exposes a FastAPI endpoint that loads the saved linear regression model from `Linear-Regression-model.pkl`, preprocesses incoming JSON, and returns a prediction.

## Project structure

```text
app/
├── main.py
├── model.py
├── preprocess.py
```

## Input schema

Send a JSON body with these fields and types:

- `Pregnancies`: int
- `Glucose`: int
- `BloodPressure`: int
- `SkinThickness`: int
- `Insulin`: int
- `BMI`: float
- `DiabetesPedigreeFunction`: float
- `Age`: int

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn app.main:app --reload
```

## Predict

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.1,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 33
  }'
```

## Notes

- The model file must stay at the project root with the name `Linear-Regression-model.pkl`.
- The API validates that the request contains only the expected fields.
- Numeric values are coerced to the required data types before prediction.
