"""KAVACH-AAYUSH demo prototype. Runs fully offline on localhost.  python main.py"""
import os, time
import numpy as np, pandas as pd, xgboost as xgb, shap
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ortools.sat.python import cp_model
try:
    from sqlcipher3 import dbapi2 as sqlite3; ENC = True      # AES-256 via SQLCipher
except ImportError:
    import sqlite3; ENC = False                                  # demo fallback: plain SQLite
BASE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(BASE, "aayush.db")
KEY = os.environ.get("AAYUSH_KEY", "change-me")
F = ["night_streak", "leave_overdue", "field_months", "weekly_hours", "pvt_ms"]
LABEL = {"night_streak": "consecutive night sentry shifts", "leave_overdue": "days overdue family leave",
         "field_months": "months continuous field tenure", "weekly_hours": "duty hours per week", "pvt_ms": "ms reaction time (PVT-B)"}
POSTS, SHIFTS, FLAG = 8, ["0000-0800", "0800-1600", "1600-2400"], 70

def db():
    c = sqlite3.connect(DB)
    if ENC: c.execute(f"PRAGMA key='{KEY}'")
    return c
def gen(n, rng):
    return pd.DataFrame({"night_streak": rng.integers(0, 8, n), "leave_overdue": rng.integers(0, 60, n),
        "field_months": rng.integers(3, 60, n), "weekly_hours": rng.integers(48, 90, n), "pvt_ms": rng.normal(300, 45, n).round()})
def z(d): return 0.5*d.night_streak + 0.045*d.leave_overdue + 0.03*d.field_months + 0.05*(d.weekly_hours-66) + 0.02*(d.pvt_ms-300) - 6.0

rng = np.random.default_rng(7)
tr = gen(4000, rng); y = (rng.random(4000) < 1/(1+np.exp(-z(tr)))).astype(int)
model = xgb.XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1).fit(tr[F], y)
explainer = shap.TreeExplainer(model)

def init_db():
    c = db(); c.execute("CREATE TABLE IF NOT EXISTS personnel(pid TEXT PRIMARY KEY,night_streak,leave_overdue,field_months,weekly_hours,pvt_ms)")
    c.execute("CREATE TABLE IF NOT EXISTS roster(slot TEXT PRIMARY KEY,pid TEXT)")
    if not c.execute("SELECT COUNT(*) FROM personnel").fetchone()[0]:
        d = gen(135, np.random.default_rng(11)); d.insert(0, "pid", [f"J-{100+i}" for i in range(135)])
        d.loc[d.pid == "J-214", F] = [5, 35, 38, 80, 382]
        c.executemany("INSERT INTO personnel VALUES(?,?,?,?,?,?)", [tuple(map(lambda v: v.item() if hasattr(v, "item") else v, r)) for r in d.itertuples(index=False)])
    c.commit(); c.close()
def people():
    c = db(); d = pd.read_sql_query("SELECT * FROM personnel", c); c.close()
    d["score"] = (model.predict_proba(d[F])[:, 1].astype(float)*100).round(1); return d
def baseline(d):
    ids = list(d.pid[:POSTS*3]); ids[3*3] = "J-214"       # J-214 starts on Bunker 4, 0000-0800
    return {f"{p}|{s}": ids[p*3+s] for p in range(POSTS) for s in range(3)}
def current_roster(d):
    c = db(); rows = dict(c.execute("SELECT slot,pid FROM roster").fetchall()); c.close()
    return rows or baseline(d)
def solve(d, base):
    t0 = time.time(); m = cp_model.CpModel(); sc = dict(zip(d.pid, d.score)); ns = dict(zip(d.pid, d.night_streak))
    ok = d[d.score < FLAG].sort_values("score"); P = sorted(set(ok.pid[:40]) | {p for p in base.values() if sc[p] < FLAG})  # standby pool + current duty
    x = {(p, p2, s): m.NewBoolVar("") for p in P for p2 in range(POSTS) for s in range(3)}
    for p2 in range(POSTS):
        for s in range(3): m.Add(sum(x[p, p2, s] for p in P) == 1)
    for p in P:
        m.Add(sum(x[p, p2, s] for p2 in range(POSTS) for s in range(3)) <= 1)
    m.Minimize(sum(x[p, p2, s]*(int(sc[p]*10) + (ns[p]*20 if s == 0 else 0) - (3000 if base[f"{p2}|{s}"] == p else 0)) for (p, p2, s) in x))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = 5; sv.Solve(m)
    new = {f"{p2}|{s}": p for (p, p2, s), v in x.items() if sv.Value(v)}
    return new, round((time.time()-t0)*1000)

app = FastAPI(title="KAVACH-AAYUSH"); init_db(); PENDING = {}
def need(role, allowed):
    if role not in allowed: raise HTTPException(403, "Role-based access: this view is not permitted for your role")
@app.get("/")
def console(): return FileResponse(os.path.join(BASE, "static/console.html"))
@app.get("/app")
def micro(): return FileResponse(os.path.join(BASE, "static/app.html"))
@app.get("/api/overview")
def overview(x_role: str = Header("commander")):
    d = people(); ros = current_roster(d); sc = dict(zip(d.pid, d.score)); risk = [p for p in ros.values() if sc[p] >= FLAG]
    out = {"role": x_role, "readiness": round(100-d.score.mean(), 1), "total": len(d), "posts": POSTS, "manned": POSTS,
           "at_risk_on_duty": len(risk), "encrypted": ENC}
    out["roster"] = [{"post": int(k.split("|")[0])+1, "shift": SHIFTS[int(k.split("|")[1])], "pid": v} for k, v in sorted(ros.items())] if x_role != "rmo" else []
    if x_role == "rmo":
        w = d[d.score >= FLAG].sort_values("score", ascending=False)
        out["watchlist"] = [{"pid": r.pid, "score": r.score} for r in w.itertuples()]
    return out
@app.get("/api/person/{pid}")
def person(pid: str, x_role: str = Header("commander")):
    need(x_role, ["rmo"]); d = people(); r = d[d.pid == pid]
    if r.empty: raise HTTPException(404, "Unknown record")
    sv = explainer.shap_values(r[F])[0]; pos = {f: max(v, 0) for f, v in zip(F, sv)}; tot = sum(pos.values()) or 1
    row = r.iloc[0]; drivers = sorted([{"feature": f, "text": f"{int(row[f])} {LABEL[f]}", "share": round(100*v/tot)} for f, v in pos.items() if v > 0], key=lambda a: -a["share"])
    act = "36-hour recovery window recommended. Proactive, non-punitive counseling arranged." if row.score >= FLAG else "Within a steady range. No action needed."
    return {"pid": pid, "score": row.score, "drivers": drivers, "action": act}
class Pvt(BaseModel): pid: str; ms: float
@app.post("/api/pvt")
def pvt(b: Pvt):
    c = db(); n = c.execute("UPDATE personnel SET pvt_ms=? WHERE pid=?", (round(b.ms), b.pid)).rowcount; c.commit(); c.close()
    if not n: raise HTTPException(404, "Unknown ID")
    return {"pid": b.pid, "ms": round(b.ms), "score": float(people().set_index("pid").loc[b.pid, "score"])}
@app.post("/api/rebalance")
def rebalance(x_role: str = Header("commander")):
    need(x_role, ["commander"]); d = people(); base = current_roster(d); new, ms = solve(d, base)
    sw = [{"post": int(k.split("|")[0])+1, "shift": SHIFTS[int(k.split("|")[1])], "out": base[k], "in": new[k]} for k in sorted(new) if new[k] != base[k]]
    PENDING["roster"] = new; return {"solver_ms": ms, "swaps": sw, "manned": POSTS, "total_slots": POSTS*3}
@app.post("/api/approve")
def approve(x_role: str = Header("commander")):
    need(x_role, ["commander"])
    if "roster" not in PENDING: raise HTTPException(400, "Run rebalance first")
    c = db(); c.execute("DELETE FROM roster"); c.executemany("INSERT INTO roster VALUES(?,?)", PENDING["roster"].items()); c.commit(); c.close()
    PENDING.clear(); return {"ok": True}
@app.post("/api/reset")
def reset():
    c = db(); c.execute("DELETE FROM roster"); c.execute("DELETE FROM personnel"); c.commit(); c.close(); init_db(); return {"ok": True}
if __name__ == "__main__":
    import uvicorn; print("Encrypted (SQLCipher):", ENC, "| open http://127.0.0.1:8000"); uvicorn.run(app, host="127.0.0.1", port=8000)
