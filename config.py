"""Shared settings used by training, the service layer and the web app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "customer_purchase.csv"
MODEL_PATH = BASE_DIR / "model" / "customer_purchase_model.pkl"
SEGMENT_PATH = BASE_DIR / "model" / "segmentation_model.pkl"
ARTEFACT_DIR = BASE_DIR / "model"          # json artefacts live here too
OUTPUT_DIR = BASE_DIR / "outputs"          # charts for the project report
DB_PATH = BASE_DIR / "history.db"
TMP_DIR = BASE_DIR / "tmp"

TARGET = "Purchased"
ID_COLUMNS = ["Customer_ID"]
NUMERIC = ["Age", "Income", "Browsing_Time", "Num_Visits"]
CATEGORICAL = ["Gender", "Previous_Purchase"]
FEATURES = NUMERIC + CATEGORICAL

LABELS = {
    "Age": "Age", "Income": "Income", "Browsing_Time": "Browsing time",
    "Num_Visits": "Number of visits", "Gender": "Gender",
    "Previous_Purchase": "Previous purchase",
}
LIMITS = {                      # accepted ranges for user input
    "Age": (10, 100), "Income": (0, 10_000_000),
    "Browsing_Time": (0, 600), "Num_Visits": (0, 1000),
}
GENDERS = ["Male", "Female"]
YES_NO = ["Yes", "No"]
