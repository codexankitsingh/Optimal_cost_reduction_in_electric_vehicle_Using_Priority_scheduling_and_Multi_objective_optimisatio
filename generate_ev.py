#!/usr/bin/env python3
"""
EV synthetic dataset generator.

Outputs CSV with columns:
  id,Ecap,SoC_init,SoC_max,R_i,T_stay,T_arr_idx,T_dep_idx,SoC_min,cdeg

Key behaviors:
 - Time discretization: delta_t = 0.25 h (15-minute slots), horizon 48 slots (24h)
 - Realistic distributions:
   • Ecap: bimodal (compact 35–55 kWh, SUV 70–100 kWh), correlated with R_i
   • SoC_init: beta distribution in [0.05, 0.7)
   • SoC_max: increment 0.20–0.45 above SoC_init, capped at 0.95 and feasibility
   • SoC_min: per-EV in [0.05, 0.20]
   • T_stay: triangular(1, 12, mode=4) hours
   • Arrival: bimodal (around 08:00 and 18:00), clamped to feasible window
 - Feasibility clamp: (SoC_max - SoC_init) * Ecap <= eta * R_i * T_stay (eta=0.95)
 - cdeg: per-EV heterogeneous (bounded log-normal around 0.02)

CLI:
  --n N           Number of EVs (default 100)
  --out PATH      Output CSV path (default evs_100.csv)
  --seed SEED     RNG seed (default 42)
"""

import csv
import math
import random
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--out", type=str, default="evs_100.csv")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

OUT = Path(args.out)
N = int(args.n)
delta_t = 0.25
HORIZON_SLOTS = 48
charger_choices = [3.6, 7.0, 11.0, 22.0]
base_charger_weights = [0.30, 0.40, 0.22, 0.08]  # baseline; will adjust by Ecap

rows = []

for i in range(1, N+1):
    # Ecap: bimodal distribution; correlate with charger power later
    if random.random() < 0.7:  # compact/sedan cluster
        Ecap = random.uniform(35.0, 55.0)
    else:  # SUV/long-range cluster
        Ecap = random.uniform(70.0, 100.0)
    Ecap = round(Ecap)

    # SoC_init: beta distribution scaled to [0.05, 0.7)
    a, b = 2.0, 5.0
    soc0 = random.betavariate(a, b)
    SoC_init = 0.05 + soc0 * (0.7 - 0.05)
    SoC_init = min(0.695, SoC_init)
    SoC_init = round(SoC_init, 3)

    # SoC_max: increment between 0.20 and 0.45, cap at 0.95
    incr = 0.20 + random.uniform(0.0, 0.25)
    SoC_max = min(0.95, SoC_init + incr)
    if SoC_max <= SoC_init:
        SoC_max = min(0.95, SoC_init + 0.21)
    SoC_max = round(SoC_max, 3)

    # T_stay: triangular skew to shorter stays, in hours [1, 12]
    T_stay = max(1.0, min(12.0, random.triangular(1.0, 12.0, 4.0)))

    # Charger rating: adjust weights upward for bigger packs
    w = list(base_charger_weights)
    if Ecap >= 70:
        # shift some probability mass to 11/22 kW
        w = [w[0]-0.05, w[1]-0.05, w[2]+0.06, w[3]+0.04]
        # re-normalize to sum to 1
        s = sum(w)
        w = [max(0.0, x)/s for x in w]
    R_i = random.choices(charger_choices, weights=w, k=1)[0]

    # Compute arrival/departure slot indices using bimodal hour-of-day
    # Peak around 08:00 and 18:00; clamp to feasible window
    slots_needed = int(math.ceil(T_stay / delta_t))
    max_arrival = HORIZON_SLOTS - slots_needed
    if max_arrival < 0:
        max_arrival = 0
    # sample hour
    if random.random() < 0.55:
        hour = random.gauss(8.0, 2.0)
    else:
        hour = random.gauss(18.0, 2.0)
    hour = max(0.0, min(24.0, hour))
    T_arr_idx = int(round(hour * 4))
    T_arr_idx = max(0, min(max_arrival, T_arr_idx))
    T_dep_idx = min(HORIZON_SLOTS, T_arr_idx + slots_needed)

    # Per-EV SoC_min
    SoC_min = round(random.uniform(0.05, 0.20), 2)
    SoC_min = min(SoC_min, SoC_init)  # ensure not above initial SoC

    # Heterogeneous cdeg: bounded log-normal around 0.02
    c_base = random.lognormvariate(mu=math.log(0.02), sigma=0.25)
    cdeg = float(min(0.05, max(0.005, c_base)))

    # Feasibility clamp for SoC_max relative to deliverable energy
    deltaE = max(0.0, (SoC_max - SoC_init) * Ecap)
    max_deliverable = R_i * T_stay
    eta = 0.95
    if deltaE > eta * max_deliverable and Ecap > 0:
        SoC_max = SoC_init + (eta * max_deliverable) / Ecap
        SoC_max = min(0.95, SoC_max)
        if SoC_max <= SoC_init:
            SoC_max = min(0.95, SoC_init + 0.01)  # minimal headroom
        SoC_max = round(SoC_max, 3)

    rows.append({
        "id": i,
        "Ecap": int(Ecap),
        "SoC_init": f"{SoC_init:.3f}",
        "SoC_max": f"{SoC_max:.3f}",
        "R_i": R_i,
        "T_stay": f"{float(T_stay):.2f}",
        "T_arr_idx": int(T_arr_idx),
        "T_dep_idx": int(T_dep_idx),
        "SoC_min": f"{SoC_min:.2f}",
        "cdeg": f"{cdeg:.4f}"
    })

# Write CSV
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id","Ecap","SoC_init","SoC_max","R_i","T_stay","T_arr_idx","T_dep_idx","SoC_min","cdeg"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} EV rows to: {OUT.resolve()}")
print("Preview (first 10 rows):")
try:
    import pandas as pd  # type: ignore
    print(pd.DataFrame(rows).head(10).to_string(index=False))
except Exception:
    for r in rows[:10]:
        print(r)
