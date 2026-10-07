# Customer Purchase Prediction Platform

A machine learning web application that predicts whether a customer will purchase,
explains the prediction, groups customers into segments and recommends a next step.

## Modules
| Module | What it does |
|---|---|
| Single prediction | Probability, segment and recommended action for one customer |
| Explainable AI | Model-agnostic local explanation: which features pushed the probability up or down |
| Customer segmentation | K-Means clustering into 4 behavioural segments |
| Batch scoring | Upload a CSV, score every row, download results |
| Analytics dashboard | Purchase patterns by age, income, browsing time, visits, etc. (Chart.js, works offline) |
| Model comparison | Grid-search tuning, 5-fold CV, ROC curves, confusion matrices, feature importance |
| Prediction history | Every prediction saved to SQLite; export or clear |
| REST API | `POST /api/predict` with JSON |

## Setup and run
```bash
pip install -r requirements.txt
python generate_dataset.py   # SYNTHETIC sample data (skip if you have a real CSV)
python train_model.py        # tune, compare, save models + analytics
python app.py                # open http://127.0.0.1:5000
```
Re-run `train_model.py` whenever the dataset changes.

## REST API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d '{"age":38,"gender":"Male","income":70000,"browsing_time":24,"previous_purchase":"Yes","num_visits":8}'
```

## Using your own dataset
1. Put your CSV at `dataset/customer_purchase.csv`.
2. Edit `config.py` (`TARGET`, `ID_COLUMNS`, `NUMERIC`, `CATEGORICAL`, `LABELS`, `LIMITS`).
3. Update the form fields in `templates/predict.html` and `FORM_FIELDS` / `build_row()` in `service.py`.
4. Run `python train_model.py`.

## Project structure
```
config.py            shared settings
generate_dataset.py  synthetic sample data
train_model.py       preprocessing, tuning, evaluation, segmentation, artefacts
service.py           prediction, explanation, recommendations, batch scoring
db.py                SQLite history
app.py               Flask routes
templates/ static/   UI (Chart.js bundled in static/vendor)
model/               saved models and JSON artefacts
outputs/             charts for the project report
```

## Important
The bundled dataset is synthetic, so the metrics are not valid for your report.
Use a real dataset and report the numbers you actually get.
