#!/usr/bin/env python3
"""
zrp_perturb.py
────────────────
Run ZRP on the full perturbation grid described in §5 (ZIP‑code noise α ∈ {0,5,10,20} %,
surname‑missing γ ∈ {0,5,10} %).  For every (α, γ) pair we write a Feather file

    artifacts/proxy_output_exp_<alpha>_<gamma>_seed<SEED>.feather

that mirrors the naming convention used by *zrp_postprocess.py*.  Existing artefacts are
skipped so the script is fully restartable.

You can override the default GRID, SEED, or output directory from the command line, e.g.

    python zrp_perturb.py --alphas 0 10 20 --gammas 0 10 \
                          --seed 42 --art-dir /tmp/zrp_runs
"""

import argparse, pathlib, random, string, sys
from itertools import product

import geopandas as gpd
import numpy  as np
import pandas as pd
from sklearn.neighbors import BallTree
from zrp import ZRP

# ───────────── CLI ──────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--alphas",  type=int, nargs="*", default=[0, 5, 10, 20],
                    help="ZIP‑code noise levels α (percent)")
parser.add_argument("--gammas", type=int, nargs="*", default=[0, 5, 10],
                    help="surname‑missing levels γ (percent)")
parser.add_argument("--seed",   type=int, default=0, help="global RNG seed")
parser.add_argument("--art-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent / "artifacts",
                    help="directory for output artefacts")
parser.add_argument("--voter-csv", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "nc_voter_cleaned_2022.csv",
                    help="cleaned voter file (CSV)")
parser.add_argument("--zcta-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "tl_2022_us_zcta520",
                    help="folder with TIGER/Line ZCTA shapefiles")
args = parser.parse_args()

# ───────────── paths / config ───────────────────────────────────
GRID       = list(product(args.alphas, args.gammas))
SEED       = args.seed
random.seed(SEED)
np.random.seed(SEED)
args.art_dir.mkdir(parents=True, exist_ok=True)

# ───────────── helpers ──────────────────────────────────────────
EARTH_MI = 3958.8                          # Earth radius in miles
LETTERS  = string.ascii_uppercase

# 3 simple string‑perturbation functions for surname corruption
def _swap_adjacent(s: str) -> str:
    if len(s) < 2:
        return s
    i = random.randint(0, len(s) - 2)
    lst = list(s)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)

def _delete_char(s: str) -> str:
    if len(s) < 2:
        return s
    i = random.randrange(len(s))
    return s[:i] + s[i + 1:]

def _add_char(s: str) -> str:
    i = random.randrange(len(s) + 1)
    return s[:i] + random.choice(LETTERS) + s[i:]

PERTURB_FN = [_swap_adjacent, _delete_char, _add_char]

# bucket mapping (race/ethnicity → evaluation bucket)

def _bucketize(r: str, e: str) -> str:
    r, e = r.upper(), e.upper()
    if r == "W":
        return "White"
    if r == "B":
        return "Black"
    if r == "A":
        return "AAPI"
    if e == "HL":
        return "Hispanic"
    return "Other"

# ───────────── build 10‑mile neighbourhood lookup ───────────────
print("• building 10‑mile BallTree …", flush=True)
shp_file = next(args.zcta_dir.glob("*.shp"))
zcta = gpd.read_file(shp_file)
for col in ("ZCTA5CE10", "ZCTA5CE20", "GEOID10", "GEOID20", "ZCTA5CE"):
    if col in zcta.columns:
        zcta = zcta.rename(columns={col: "zip"})
        break
# only NC ZIPs (27***)
zcta = zcta[zcta["zip"].str.startswith("27")].reset_index(drop=True)
zcta["centroid"] = zcta.geometry.centroid
coords = np.vstack([zcta.centroid.y.values, zcta.centroid.x.values]).T
# Haversine distance on the sphere (radians)
tree = BallTree(np.deg2rad(coords), metric="haversine")

# map ZIP → list of neighbour ZIPs within 10 miles

def _neigh_10mi(zip_code: str):
    row = zcta.loc[zcta["zip"] == zip_code]
    if row.empty:
        return [zip_code]
    idx = row.index[0]
    ind = tree.query_radius(coords[[idx]], r=10 / EARTH_MI)[0]
    return zcta.iloc[ind]["zip"].tolist()

ZIP_NEIGHB = {z: _neigh_10mi(z) for z in zcta["zip"]}

def _perturb_zip(z: str) -> str:
    cands = [x for x in ZIP_NEIGHB.get(z, [z]) if x != z]
    return random.choice(cands) if cands else z

# ───────────── load and precompute voter dataframe ──────────────
print("• loading voter CSV …", flush=True)
voters = pd.read_csv(args.voter_csv, dtype={"zip_code": str})
voters["bucket"]   = voters.apply(lambda r: _bucketize(r["race_code"], r["ethnic_code"]), axis=1)
voters["ZEST_KEY"] = voters["ncid"].astype(str)

# balanced subsampling mask per evaluation bucket  citeturn2file8

def _balanced_mask(df: pd.DataFrame, pct: int) -> np.ndarray:
    m = np.zeros(len(df), bool)
    for _, grp in df.groupby("bucket", observed=True):
        k = int(len(grp) * pct / 100)
        if k:
            m[np.random.choice(grp.index, k, False)] = True
    return m

# ───────────── iterate over perturbation grid ───────────────────
for alpha, gamma in GRID:
    artefact = args.art_dir / f"proxy_output_exp_{alpha}_{gamma}_seed{SEED}.feather"
    if artefact.exists():
        print(f"✓ artefact for α={alpha}% γ={gamma}% already present – skip")
        continue

    print(f"\n▶ running α={alpha}% ZIP noise | γ={gamma}% surname noise …", flush=True)
    df = voters.copy()

    # ZIP‑code noise
    if alpha:
        m = _balanced_mask(df, alpha)
        df.loc[m, "zip_code"] = df.loc[m, "zip_code"].apply(_perturb_zip)

    # surname corruption (missing token)
    if gamma:
        m = _balanced_mask(df, gamma)
        df.loc[m, "surname"] = (
            df.loc[m, "surname"]
              .str.upper()
              .apply(lambda s: random.choice(PERTURB_FN)(s))
        )

    # prepare ZRP input frame
    zrp_in = pd.DataFrame({
        "first_name"    : df["first"],
        "last_name"     : df["surname"],
        "state"         : "NC",
        "zip_code"      : df["zip_code"].str.zfill(5),
        "ZEST_KEY"      : df["ZEST_KEY"],
        "middle_name"   : "",
        "house_number"  : "",
        "street_address": "",
    })

    # run ZRP (geocode & BISG disabled, consistent with prior scripts)  citeturn2file2
    zest = ZRP(geocode=False, bisg=False, runname=f"exp_{alpha}_{gamma}_seed{SEED}")
    zest.fit()
    preds = zest.transform(zrp_in)
    preds.columns = preds.columns.str.upper()

    preds.to_feather(artefact)
    print(f"  ↳ artefact written to {artefact.relative_to(args.art_dir.parent)}", flush=True)

print("\n✅ all requested perturbations finished.")
