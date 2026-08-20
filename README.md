# Credit default classifier

Binary classifier predicting credit card default. Trained on a synthetic dataset matching the structure of the UCI Taiwan Credit Card Default dataset (30,000-client cohort). Served via FastAPI.

## Stack

- scikit-learn — preprocessing pipeline, logistic regression, random forest, gradient boosting
- FastAPI + pydantic — REST inference endpoint
- SHAP — per-request feature attribution
- Docker + Railway — containerized deployment

## Results

| Model | CV ROC-AUC |
|---|---|
| Logistic regression | TBD |
| Random forest | TBD |
| Gradient boosting | TBD |

Plots and final metrics added after training.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload
```

## API

```
POST /predict
Content-Type: application/json

{
  "limit_bal": 50000,
  "age": 35,
  "education": 2,
  "marriage": 1,
  "pay_0": 0,
  ...
}
```

Response includes `default_probability`, `prediction`, and `shap_values`.

## Train

```bash
python src/train.py
```

Outputs trained model to `models/best_model.joblib` and plots to `plots/`.

## Test

```bash
pytest tests/
```
