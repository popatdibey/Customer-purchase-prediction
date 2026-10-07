"""Train, tune, compare and save models plus every analytics artefact the web app uses.

Run:  python train_model.py
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (ARTEFACT_DIR, CATEGORICAL, DATA_PATH, FEATURES, ID_COLUMNS, LABELS,
                    MODEL_PATH, NUMERIC, OUTPUT_DIR, SEGMENT_PATH, TARGET)
from service import segment_frame

ARTEFACT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_json(obj, name):
    with open(ARTEFACT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================ 1. load + clean
data = pd.read_csv(DATA_PATH)
print("Shape:", data.shape)
print("Missing values:\n", data.isnull().sum()[lambda s: s > 0].to_string())
data = data.drop(columns=[c for c in ID_COLUMNS if c in data.columns]).drop_duplicates()
X, y = data[FEATURES], data[TARGET]

# ============================================================ 2. dashboard analytics
def rate_by(groups):
    g = data.groupby(groups, observed=True)[TARGET].agg(["mean", "size"])
    return {"labels": [str(i) for i in g.index],
            "values": [round(float(v) * 100, 1) for v in g["mean"]],
            "counts": [int(c) for c in g["size"]]}

dash = {
    "kpis": {
        "customers": int(len(data)),
        "purchase_rate": round(float(y.mean()) * 100, 1),
        "avg_income": int(data["Income"].mean()),
        "avg_age": round(float(data["Age"].mean()), 1),
        "avg_browsing": round(float(data["Browsing_Time"].mean()), 1),
    },
    "age": rate_by(pd.cut(data["Age"], [0, 25, 35, 45, 55, 200],
                          labels=["up to 25", "26-35", "36-45", "46-55", "56+"])),
    "income": rate_by(pd.cut(data["Income"], [0, 30000, 50000, 70000, 90000, 1e9],
                             labels=["<30k", "30-50k", "50-70k", "70-90k", "90k+"])),
    "visits": rate_by(pd.cut(data["Num_Visits"], [0, 3, 6, 9, 1000],
                             labels=["1-3", "4-6", "7-9", "10+"])),
    "browsing": rate_by(pd.cut(data["Browsing_Time"], [0, 8, 15, 25, 1000],
                               labels=["<8 min", "8-15", "15-25", "25+ min"])),
    "previous": rate_by(data["Previous_Purchase"]),
    "gender": rate_by(data["Gender"]),
}

# static EDA figure for the report
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(data=data, x="Age", hue=TARGET, multiple="stack", ax=ax[0, 0]); ax[0, 0].set_title("Age vs Purchase")
sns.boxplot(data=data, x=TARGET, y="Income", ax=ax[0, 1]); ax[0, 1].set_title("Income vs Purchase")
sns.countplot(data=data, x="Previous_Purchase", hue=TARGET, ax=ax[1, 0]); ax[1, 0].set_title("Previous Purchase vs Purchase")
sns.heatmap(data[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax[1, 1]); ax[1, 1].set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "eda.png", dpi=130); plt.close()

# ============================================================ 3. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================ 4. tune + compare models
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])
search_space = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"model__C": [0.1, 1, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"model__n_estimators": [100, 200], "model__max_depth": [None, 6, 10]}),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models, fitted = [], {}
for name, (clf, grid) in search_space.items():
    gs = GridSearchCV(Pipeline([("prep", preprocess), ("model", clf)]), grid,
                      cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    pipe = gs.best_estimator_
    pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    models.append({
        "name": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "cv_auc_mean": round(float(gs.best_score_), 4),
        "cv_auc_std": round(float(gs.cv_results_["std_test_score"][gs.best_index_]), 4),
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": {"fpr": [round(float(v), 3) for v in fpr[keep]],
                "tpr": [round(float(v), 3) for v in tpr[keep]]},
    })
    fitted[name] = pipe

table = pd.DataFrame(models).set_index("name")[
    ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean", "cv_auc_std"]]
print("\n=== Tuned model comparison (hold-out test set) ===\n", table.round(3).to_string())

# choose on cross-validation (not on the test set) to avoid leakage
best_name = max(models, key=lambda m: m["cv_auc_mean"])["name"]
best = fitted[best_name]
print("\nBest model (highest CV ROC-AUC):", best_name)

# ============================================================ 5. feature importance
perm = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                              n_repeats=20, random_state=42)
importance = sorted(
    [{"feature": LABELS[f], "importance": round(float(m), 4)}
     for f, m in zip(FEATURES, perm.importances_mean)],
    key=lambda d: d["importance"], reverse=True)

# ============================================================ 6. report figures
table[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
    kind="bar", figsize=(9, 4.5), ylim=(0, 1))
plt.title("Model comparison"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=130); plt.close()
table.round(4).to_csv(OUTPUT_DIR / "model_comparison.csv")

plt.figure(figsize=(5.5, 5))
for m in models:
    plt.plot(m["roc"]["fpr"], m["roc"]["tpr"], label=f'{m["name"]} (AUC {m["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=130); plt.close()

cm = confusion_matrix(y_test, best.predict(X_test))
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=130); plt.close()

plt.figure(figsize=(6, 3.5))
plt.barh([i["feature"] for i in importance][::-1], [i["importance"] for i in importance][::-1], color="#1f5f8b")
plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title("Feature importance (permutation)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=130); plt.close()

# ============================================================ 7. customer segmentation
seg_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("km", KMeans(n_clusters=4, n_init=10, random_state=42))])
seg_X = segment_frame(data)
labels = seg_pipe.fit_predict(seg_X)

PHRASES = {"Age": ("older", "younger"), "Income": ("higher income", "lower income"),
           "Browsing_Time": ("longer sessions", "shorter sessions"),
           "Num_Visits": ("frequent visitors", "infrequent visitors")}
rates = {c: float(data[TARGET][labels == c].mean()) for c in range(4)}
rank = sorted(rates, key=rates.get)                      # lowest -> highest purchase rate
SEG_NAMES = ["Window Shoppers", "Casual Browsers", "Engaged Prospects", "High-Intent Buyers"]

mu, sd = seg_X.mean(), seg_X.std()
segments = {}
for c in range(4):
    sub = seg_X[labels == c]
    z = ((sub.mean() - mu) / sd).sort_values(key=abs, ascending=False)[:2]
    desc = ", ".join(PHRASES[f][0 if v > 0 else 1] for f, v in z.items())
    segments[str(c)] = {
        "id": c, "name": SEG_NAMES[rank.index(c)], "description": desc.capitalize(),
        "size": int(len(sub)), "share": round(100 * len(sub) / len(data), 1),
        "purchase_rate": round(rates[c] * 100, 1),
        "avg_age": round(float(sub["Age"].mean()), 1),
        "avg_income": int(sub["Income"].mean()),
        "avg_browsing": round(float(sub["Browsing_Time"].mean()), 1),
        "avg_visits": round(float(sub["Num_Visits"].mean()), 1),
        "repeat_pct": round(float((data["Previous_Purchase"][labels == c] == "Yes").mean()) * 100, 1),
    }
dash["segments"] = sorted(segments.values(), key=lambda s: s["purchase_rate"])
sample = seg_X.assign(seg=[segments[str(l)]["name"] for l in labels]).dropna().sample(350, random_state=1)
dash["scatter"] = [{"x": float(r.Income), "y": float(r.Browsing_Time), "s": r.seg}
                   for r in sample.itertuples()]
print("\nSegments:")
for s in dash["segments"]:
    print(f'  {s["name"]:<18} n={s["size"]:<5} purchase={s["purchase_rate"]:>5}%  {s["description"]}')

# ============================================================ 8. save artefacts
meta = {
    "baselines": {**{c: float(X_train[c].median()) for c in NUMERIC},
                  **{c: X_train[c].mode()[0] for c in CATEGORICAL}},
}
joblib.dump(best, MODEL_PATH)
joblib.dump(seg_pipe, SEGMENT_PATH)
save_json(meta, "meta.json")
save_json(segments, "segments.json")
save_json(dash, "dashboard.json")
save_json({"best": best_name, "models": models, "importance": importance,
           "split": {"train": int(len(X_train)), "test": int(len(X_test))},
           "cv_folds": 5}, "metrics.json")
print("\nSaved model + analytics artefacts.")
