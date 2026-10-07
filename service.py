"""ML service layer: prediction, explanation, segmentation, recommendations."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import (ARTEFACT_DIR, FEATURES, GENDERS, LABELS, LIMITS, MODEL_PATH,
                    NUMERIC, SEGMENT_PATH, YES_NO)

FORM_FIELDS = {
    "age": "Age", "gender": "Gender", "income": "Income",
    "browsing_time": "Browsing_Time", "previous_purchase": "Previous_Purchase",
    "num_visits": "Num_Visits",
}
REQUIRED_CSV_COLUMNS = list(FORM_FIELDS.values())


def segment_frame(df):
    """Feature frame used by the customer-segmentation (K-Means) model."""
    return df[NUMERIC].apply(pd.to_numeric, errors="coerce").copy()


def _load_json(name):
    with open(ARTEFACT_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def artefacts():
    return {
        "model": joblib.load(MODEL_PATH),
        "segmenter": joblib.load(SEGMENT_PATH),
        "meta": _load_json("meta.json"),
        "segments": _load_json("segments.json"),
        "metrics": _load_json("metrics.json"),
        "dashboard": _load_json("dashboard.json"),
    }


# ------------------------------------------------------------ input handling
def build_row(raw):
    """Validate one customer (form/JSON keys in lower_snake_case) -> DataFrame."""
    try:
        row = {
            "Age": int(float(raw["age"])),
            "Income": float(raw["income"]),
            "Browsing_Time": float(raw["browsing_time"]),
            "Num_Visits": int(float(raw["num_visits"])),
            "Gender": str(raw["gender"]).strip().title(),
            "Previous_Purchase": str(raw["previous_purchase"]).strip().title(),
        }
    except (TypeError, ValueError):
        raise ValueError("Every numeric field needs a valid number.")
    for col, (lo, hi) in LIMITS.items():
        if not lo <= row[col] <= hi:
            raise ValueError(f"{LABELS[col]} must be between {lo:,} and {hi:,}.")
    if row["Gender"] not in GENDERS:
        raise ValueError("Gender must be Male or Female.")
    if row["Previous_Purchase"] not in YES_NO:
        raise ValueError("Previous purchase must be Yes or No.")
    return pd.DataFrame([row])[FEATURES]


def _fmt(feature, value):
    if feature == "Income":
        return f"₹{float(value):,.0f}"
    if feature == "Browsing_Time":
        return f"{float(value):g} min"
    if feature in ("Age", "Num_Visits"):
        return f"{float(value):.0f}"
    return str(value)


# ------------------------------------------------------------ explanations
def explain(row):
    """Model-agnostic local explanation.

    Replace each feature by the typical (baseline) value and measure how much the
    purchase probability changes. A positive impact means the customer's actual
    value pushes the probability UP compared with an average customer.
    """
    a = artefacts()
    base = a["meta"]["baselines"]
    actual = row.iloc[0].to_dict()
    variants = [actual]
    for f in FEATURES:
        v = dict(actual)
        v[f] = base[f]
        variants.append(v)
    probs = a["model"].predict_proba(pd.DataFrame(variants)[FEATURES])[:, 1]
    items = [{
        "feature": f, "label": LABELS[f], "value": _fmt(f, actual[f]),
        "impact": round(float(probs[0] - probs[i + 1]) * 100, 1),
    } for i, f in enumerate(FEATURES)]
    items.sort(key=lambda d: abs(d["impact"]), reverse=True)
    scale = max(max(abs(i["impact"]) for i in items), 1.0)
    for i in items:
        i["width"] = round(abs(i["impact"]) / scale * 50, 1)
    return items, float(probs[0])


TIPS = {
    "Browsing_Time": "Short sessions are holding this customer back. Improve product pages and show related items to keep them engaged.",
    "Num_Visits": "Few visits so far. A re-engagement email or push notification can bring them back.",
    "Previous_Purchase": "First-time buyer. A first-order incentive such as free shipping lowers the barrier.",
    "Income": "Likely price-sensitive. Highlight EMI options, budget ranges or bundles.",
    "Age": "Age profile is less favourable. Personalise content and recommendations for this age group.",
    "Gender": "Personalise the product mix shown to this customer.",
}


def recommend(prob, impacts):
    if prob >= 0.7:
        tier, action = "Hot lead", "Nudge towards checkout with a cart reminder or free-shipping offer. A deep discount is not needed."
    elif prob >= 0.4:
        tier, action = "Warm lead", "Send a targeted discount or product bundle and retarget within 48 hours."
    else:
        tier, action = "Cold lead", "Use low-cost nurturing such as newsletters and product education. Hold back paid campaigns for now."
    negatives = [i for i in impacts if i["impact"] < -2]
    tips = [TIPS[i["feature"]] for i in negatives[:2]]
    return {"tier": tier, "action": action, "tips": tips}


def priority(prob):
    return "Hot" if prob >= 0.7 else "Warm" if prob >= 0.4 else "Cold"


# ------------------------------------------------------------ segmentation
def segments_for(df):
    a = artefacts()
    ids = a["segmenter"].predict(segment_frame(df))
    return [a["segments"][str(i)] for i in ids]


def analyse(row):
    """Full analysis of one customer."""
    impacts, p = explain(row)
    seg = segments_for(row)[0]
    return {
        "inputs": row.iloc[0].to_dict(),
        "probability": round(p * 100, 1),
        "likely": p >= 0.5,
        "label": "Likely to purchase" if p >= 0.5 else "Unlikely to purchase",
        "segment": {"name": seg["name"], "description": seg["description"]},
        "explain": impacts,
        "advice": recommend(p, impacts),
    }


# ------------------------------------------------------------ batch scoring
def predict_batch(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Gender", "Previous_Purchase"):
        df[c] = df[c].astype(str).str.strip().str.title()
    valid = df["Gender"].isin(GENDERS) & df["Previous_Purchase"].isin(YES_NO)
    for c, (lo, hi) in LIMITS.items():
        valid &= df[c].isna() | df[c].between(lo, hi)   # blanks are imputed by the model
    skipped = int((~valid).sum())
    out = df[valid].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows found. Check the column values against the template.")
    probs = artefacts()["model"].predict_proba(out[FEATURES])[:, 1]
    out["Probability_%"] = np.round(probs * 100, 1)
    out["Prediction"] = np.where(probs >= 0.5, "Likely", "Unlikely")
    out["Priority"] = [priority(p) for p in probs]
    out["Segment"] = [s["name"] for s in segments_for(out)]
    return out, skipped
