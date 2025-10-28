#!/usr/bin/env python3
"""
Generate evs_100.csv — 100 EV rows following the format:
id,Ecap,SoC_init,SoC_max,R_i,T_stay,T_arr_idx,T_dep_idx,SoC_min,cdeg

Rules enforced:
 - SoC_init in [0.05, 0.7), SoC_max > SoC_init and <= 0.95
 - R_i in {3.6, 7.0, 11.0, 22.0}
 - T_stay integer hours in [1,12]
 - T_arr_idx in [0, 47], T_dep_idx = min(48, T_arr_idx + slots_needed)
 - slots_needed = ceil(T_stay / delta_t), with delta_t = 0.25 (15-minute slots)
 - cdeg = 0.02, SoC_min = 0.0
"""

import csv
import math
import random
from pathlib import Path

random.seed(42)

OUT = Path("evs_100.csv")
N = 100
delta_t = 0.25
HORIZON_SLOTS = 48
charger_choices = [3.6, 7.0, 11.0, 22.0]
charger_weights = [0.25, 0.4, 0.25, 0.10]  # more mid-range chargers likelihood

rows = []

for i in range(1, N+1):
    # Ecap: cycle through 30..100 in steps of 5 (keeps values realistic)
    Ecap = 30 + ((i-1) % 15) * 5  # values: 30,35,...,100, then repeat

    # SoC_init: base pattern across vehicles but small random jitter
    base_init = 0.1 + ((i-1) % 7) * 0.08  # cycles 0.10,0.18,... up to <0.7
    SoC_init = round(min(0.7, base_init + random.uniform(0.0, 0.03)), 3)

    # SoC_max: SoC_init + between 0.20 and 0.45, capped at 0.95
    SoC_max = round(min(0.95, SoC_init + 0.20 + random.uniform(0.0, 0.25)), 3)
    if SoC_max <= SoC_init:
        SoC_max = round(min(0.95, SoC_init + 0.21), 3)

    # Charger rating
    R_i = random.choices(charger_choices, weights=charger_weights, k=1)[0]

    # T_stay: 1..12 hours cycle
    T_stay = 1 + ((i-1) % 12)

    # Compute arrival/departure slot indices. Ensure T_dep_idx <= 48
    slots_needed = int(math.ceil(T_stay / delta_t))
    max_arrival = HORIZON_SLOTS - slots_needed
    if max_arrival < 0:
        max_arrival = 0
    T_arr_idx = random.randint(0, max_arrival)
    T_dep_idx = min(HORIZON_SLOTS, T_arr_idx + slots_needed)

    SoC_min = 0.0
    cdeg = 0.02

    rows.append({
        "id": i,
        "Ecap": int(Ecap),
        "SoC_init": f"{SoC_init:.3f}",
        "SoC_max": f"{SoC_max:.3f}",
        "R_i": R_i,
        "T_stay": f"{float(T_stay):.2f}",
        "T_arr_idx": int(T_arr_idx),
        "T_dep_idx": int(T_dep_idx),
        "SoC_min": f"{SoC_min:.1f}",
        "cdeg": f"{cdeg:.2f}"
    })

# Write CSV
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id","Ecap","SoC_init","SoC_max","R_i","T_stay","T_arr_idx","T_dep_idx","SoC_min","cdeg"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} EV rows to: {OUT.resolve()}")
print("Preview (first 10 rows):")
import pandas as pd
print(pd.DataFrame(rows).head(10).to_string(index=False))
