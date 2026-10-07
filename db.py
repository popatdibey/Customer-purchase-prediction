"""SQLite prediction history."""
import sqlite3
from contextlib import closing

from config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with closing(_conn()) as c, c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source TEXT, age INTEGER, gender TEXT, income REAL,
            browsing_time REAL, previous_purchase TEXT, num_visits INTEGER,
            probability REAL, prediction TEXT, segment TEXT)""")


def log(source, r):
    i = r["inputs"]
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO predictions (source, age, gender, income, browsing_time, "
            "previous_purchase, num_visits, probability, prediction, segment) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (source, i["Age"], i["Gender"], i["Income"], i["Browsing_Time"],
             i["Previous_Purchase"], i["Num_Visits"], r["probability"],
             "Likely" if r["likely"] else "Unlikely", r["segment"]["name"]))


def recent(limit=100):
    with closing(_conn()) as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]


def stats():
    with closing(_conn()) as c:
        r = c.execute("SELECT COUNT(*) n, AVG(probability) avg_p, "
                      "SUM(prediction='Likely') likely FROM predictions").fetchone()
    n = r["n"] or 0
    return {"total": n, "avg_prob": round(r["avg_p"] or 0, 1),
            "likely_pct": round(100 * (r["likely"] or 0) / n, 1) if n else 0}


def clear():
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM predictions")
