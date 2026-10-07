"""Flask web application: Customer Purchase Prediction platform."""
import io
import re
import uuid

import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request, send_file,
                   url_for)

import db
import service
from config import TMP_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024      # 5 MB upload limit
TMP_DIR.mkdir(exist_ok=True)
db.init()


def best_model_info():
    m = service.artefacts()["metrics"]
    best = next(x for x in m["models"] if x["name"] == m["best"])
    return {"name": best["name"], "accuracy": round(best["accuracy"] * 100, 1)}


@app.context_processor
def inject_globals():
    return {"best_model": best_model_info()}


# ------------------------------------------------------------- 1. single prediction
@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    values = {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = service.analyse(service.build_row(values))
            db.log("Form", result)
        except KeyError:
            error = "Please fill in every field."
        except ValueError as e:
            error = str(e)
    return render_template("predict.html", result=result, error=error, values=values)


# ------------------------------------------------------------- 2. batch prediction
@app.route("/batch", methods=["GET", "POST"])
def batch():
    summary = error = None
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            error = "Choose a .csv file to upload."
        else:
            try:
                out, skipped = service.predict_batch(pd.read_csv(f))
            except ValueError as e:
                error = str(e)
            except Exception:
                error = "That file couldn't be read. Make sure it is a valid CSV."
            else:
                rid = uuid.uuid4().hex
                out.to_csv(TMP_DIR / f"{rid}.csv", index=False)
                summary = {
                    "rid": rid, "rows": len(out), "skipped": skipped,
                    "likely": int((out["Prediction"] == "Likely").sum()),
                    "hot": int((out["Priority"] == "Hot").sum()),
                    "avg": round(float(out["Probability_%"].mean()), 1),
                    "segments": out["Segment"].value_counts().to_dict(),
                    "columns": list(out.columns),
                    "preview": out.head(15).values.tolist(),
                }
    return render_template("batch.html", summary=summary, error=error)


@app.route("/batch/download/<rid>")
def batch_download(rid):
    if not re.fullmatch(r"[0-9a-f]{32}", rid) or not (TMP_DIR / f"{rid}.csv").exists():
        return redirect(url_for("batch"))
    return send_file(TMP_DIR / f"{rid}.csv", as_attachment=True,
                     download_name="purchase_predictions.csv")


@app.route("/batch/template")
def batch_template():
    sample = ("Customer_ID,Age,Gender,Income,Browsing_Time,Previous_Purchase,Num_Visits\n"
              "C001,38,Male,70000,24,Yes,8\n"
              "C002,24,Female,28000,8,No,3\n"
              "C003,36,Female,60000,22,No,7\n")
    return send_file(io.BytesIO(sample.encode()), as_attachment=True,
                     download_name="batch_template.csv", mimetype="text/csv")


# ------------------------------------------------------------- 3. analytics pages
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", d=service.artefacts()["dashboard"])


@app.route("/models")
def models():
    return render_template("models.html", m=service.artefacts()["metrics"])


@app.route("/about")
def about():
    return render_template("about.html")


# ------------------------------------------------------------- 4. history
@app.route("/history")
def history():
    return render_template("history.html", rows=db.recent(100), stats=db.stats())


@app.route("/history/export")
def history_export():
    df = pd.DataFrame(db.recent(100000))
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return send_file(buf, as_attachment=True, download_name="prediction_history.csv",
                     mimetype="text/csv")


@app.route("/history/clear", methods=["POST"])
def history_clear():
    db.clear()
    return redirect(url_for("history"))


# ------------------------------------------------------------- 5. REST API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """POST JSON: {age, gender, income, browsing_time, previous_purchase, num_visits}"""
    try:
        result = service.analyse(service.build_row(request.get_json(silent=True) or {}))
    except KeyError as e:
        return jsonify(error=f"Missing field: {e.args[0]}"), 400
    except ValueError as e:
        return jsonify(error=str(e)), 400
    db.log("API", result)
    return jsonify(
        prediction="Likely" if result["likely"] else "Unlikely",
        probability_percent=result["probability"],
        segment=result["segment"]["name"],
        priority=result["advice"]["tier"],
        recommended_action=result["advice"]["action"],
        factors=[{"feature": e["label"], "impact_points": e["impact"]}
                 for e in result["explain"]],
    )


if __name__ == "__main__":
    app.run(debug=True)
