#!/usr/bin/env python3
"""
zrp_postprocess.py
──────────────────
Compute metrics for every finished ZRP experiment artefact that
includes a *seed* in the file-name, e.g.

    artifacts/proxy_output_exp_5_0_seed0.feather

Outputs a summary CSV  →  zrp_metrics_sofar.csv
"""

import pathlib, re, sys
import numpy as np
import pandas as pd
from   sklearn.metrics import accuracy_score, f1_score, log_loss

# ───────────── paths / config ─────────────────────────────────────
HERE         = pathlib.Path(__file__).resolve().parent
ART_DIR      = HERE / "artifacts"
VOTER_CSV    = HERE / "nc_voter_cleaned_2022.csv"
OUT_CSV      = HERE / "zrp_metrics_sofar3.csv"

BUCKETS   = ["WHITE", "BLACK", "HISPANIC", "AAPI", "OTHER"]
LABEL2IDX = dict(White=0, Black=1, Hispanic=2, AAPI=3, Other=4)

# ───────── helpers ───────────────────────────────────────────────
def bucketize(row):
    r, e = row["race_code"].upper(), row["ethnic_code"].upper()
    if r == "W": return "White"
    if r == "B": return "Black"
    if r == "A": return "AAPI"
    if e == "HL": return "Hispanic"
    return "Other"

def ece10(p, y):
    conf = p.max(1)
    pred = p.argmax(1)
    acc  = (pred == y).astype(float)
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return ece

def evaluate(df):
    # rename AIAN → OTHER if present
    if "AIAN" in df.columns and "OTHER" not in df.columns:
        df = df.rename(columns={"AIAN": "OTHER"})
    # make sure all five prob columns exist
    for col in BUCKETS:
        if col not in df.columns:
            df[col] = np.nan

    ok   = df[BUCKETS].notna().all(1)
    y_p  = df.loc[ok, BUCKETS].to_numpy(float)
    y_p /= y_p.sum(1, keepdims=True)
    y_t  = df.loc[ok, "bucket"].map(LABEL2IDX).to_numpy()

    return dict(
        coverage = ok.mean(),
        accuracy = accuracy_score(y_t, y_p.argmax(1)),
        macro_f1 = f1_score     (y_t, y_p.argmax(1), average="macro"),
        logloss  = log_loss     (y_t, y_p),
        ece10    = ece10(y_p, y_t),
        n_dropped= len(df) - ok.sum()
    )

# ───────── ground-truth voter data ───────────────────────────────
print("→ loading voter CSV …")
voters = pd.read_csv(VOTER_CSV, dtype={"zip_code": str})
voters["bucket"]   = voters.apply(bucketize, axis=1)
voters["ZEST_KEY"] = voters["ncid"].astype(str)
voters             = voters[["ZEST_KEY", "bucket"]]

# ───────── iterate over artefacts ─────────────────────────────────
rows = []
pat  = re.compile(r"proxy_output_exp_(\d+)_(\d+)_seed(\d+)\.feather$")

for f in ART_DIR.glob("proxy_output_exp_*_seed*.feather"):
    m = pat.search(f.name)
    if not m:
        continue
    alpha, gamma, seed = map(int, m.groups())
    print(f"• metrics for α={alpha}%, γ={gamma}%, seed={seed} …", end="", flush=True)

    preds = pd.read_feather(f)
    preds.columns = preds.columns.str.upper()
    merged = voters.merge(preds, on="ZEST_KEY", how="inner")

    stats = evaluate(merged)
    stats.update(dict(alpha=alpha, gamma=gamma, seed=seed))
    rows.append(stats)
    print("done.")

if not rows:
    sys.exit(f"No artefacts found in {ART_DIR}/ .")

(pd.DataFrame(rows)
   .sort_values(["alpha", "gamma", "seed"])
   .to_csv(OUT_CSV, index=False))

print(f"\n✅  Wrote metrics for {len(rows)} experiment(s) to {OUT_CSV}")