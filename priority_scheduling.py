# mtp project part 1
# priority wala part

from typing import List, Dict, Tuple

# -------------------------
# Helper functions (Eq. refs in comments)
# -------------------------

def calc_delta_E(ev: Dict) -> float:
    """Eq. (7): Delta E_i = (SoC_max - SoC_init) * Ecap (kWh)"""
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    """
    Eq. (9): p_req = DeltaE / T_stay (kW)
    --- To reproduce the paper's example numbers, we cap required
    average power at the charger rating R_i (P_ref) when forming the
    'practical' required power used in aggregate demand (paper's example uses
    preq for EV2 = 11 kW instead of raw 12 kW).
    """
    deltaE = calc_delta_E(ev)
    if ev['T_stay'] <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / ev['T_stay']
    # cap at charger rating to match paper's numerical example
    return min(raw, ev['R_i'])

def calc_phi(ev: Dict) -> float:
    """
    Eq. (10) (urgency factor): phi = min(1, p_req / P_ref)
    Note: paper's example uses phi1=1.0, phi2=1.0, phi3≈0.476
    """
    p_req = calc_p_req(ev)
    pref = ev['R_i']  # P_ref is charger rated power R_i as used in example
    if pref <= 0:
        return 0.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], cdeg: float) -> Dict[int, float]:
    """
    Eq. (23): D_factor_i = (cdeg * DeltaE_i - min_j(cdeg * DeltaE_j)) / (max_j(...) - min_j(...))
    Returns dict mapping ev_id -> Dfactor
    """
    values = []
    for ev in ev_list:
        dval = cdeg * calc_delta_E(ev)
        values.append(dval)
    vmin = min(values)
    vmax = max(values)
    denom = vmax - vmin if vmax != vmin else 1.0  # avoid zero division
    dfactor = {}
    for ev, val in zip(ev_list, values):
        dfactor[ev['id']] = (val - vmin) / denom
    return dfactor

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    """
    Eq. (24): G_factor_t = max(0, (P_hat_agg - P_avg) / (P_max - P_avg))
    where P_hat_agg = sum_j p_req_j (using capped preq as per example)
    """
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = P_max - P_avg if P_max != P_avg else 1.0
    return max(0.0, (p_hat - P_avg) / denom)

def calc_price_factor(pi_buy: float, pi_rev: float,
                      pi_buy_min: float, pi_buy_max: float,
                      pi_rev_min: float, pi_rev_max: float) -> float:
    """
    Eq. (25)-(26):
    P_buy = (pi_buy - pi_buy_min) / (pi_buy_max - pi_buy_min)
    P_rev = (pi_rev - pi_rev_min) / (pi_rev_max - pi_rev_min)
    P_factor = P_buy - P_rev
    """
    # normalize, protecting against zero ranges
    buy_denom = (pi_buy_max - pi_buy_min) if (pi_buy_max - pi_buy_min) != 0 else 1.0
    rev_denom = (pi_rev_max - pi_rev_min) if (pi_rev_max - pi_rev_min) != 0 else 1.0
    P_buy = (pi_buy - pi_buy_min) / buy_denom
    P_rev = (pi_rev - pi_rev_min) / rev_denom
    return P_buy - P_rev

def calc_priority_scores(ev_list: List[Dict], weights: Dict[str, float],
                         cdeg: float, P_avg: float, P_max: float,
                         pi_buy: float, pi_rev: float,
                         pi_buy_min: float, pi_buy_max: float,
                         pi_rev_min: float, pi_rev_max: float) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    """
    Compute phi, Dfactor, Gfactor, Pfactor and lambda for each EV.
    Returns:
      - lambda_scores: dict ev_id -> lambda
      - details: dict ev_id -> {phi, Dfactor, ...}
    """
    # urgency phi per EV
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}

    # degradation factor (normalized)
    D_map = calc_degradation_factor(ev_list, cdeg)

    # grid stress factor (same for all EVs)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)

    # price factor (same for all EVs)
    P_factor = calc_price_factor(pi_buy, pi_rev,
                                 pi_buy_min, pi_buy_max,
                                 pi_rev_min, pi_rev_max)

    # compute lambda (Eq. (22))
    lambda_scores = {}
    details = {}
    for ev in ev_list:
        eid = ev['id']
        phi = phi_map[eid]
        D = D_map[eid]
        lam = (weights['w_s'] * phi
               - weights['w_d'] * D
               - weights['w_g'] * G_factor
               - weights['w_p'] * P_factor)
        lambda_scores[eid] = lam
        details[eid] = {
            'phi': phi,
            'Dfactor': D,
            'Gfactor': G_factor,
            'Pfactor': P_factor,
            'lambda': lam,
            'p_req': calc_p_req(ev),
            'DeltaE': calc_delta_E(ev)
        }
    return lambda_scores, details

def assign_chargers(lambda_scores: Dict[int, float], M: int) -> List[int]:
    """
    Rank EVs by descending lambda. Tie-breaker: lower EV id first.
    Return list of admitted EV ids (top M).
    """
    # sort by (-lambda, id)
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    admitted = [eid for eid, _ in sorted_evs[:M]]
    return admitted

# -------------------------
# Main orchestration
# -------------------------

def main():
    # System parameters (from numerical example) - see paper Table I & description. :contentReference[oaicite:2]{index=2}
    N = 3
    M = 2
    P_max = 25.0  # kW
    P_avg = 12.0  # kW
    cdeg = 0.02   # $/kWh
    pi_buy = 0.25
    pi_rev = 0.18
    pi_buy_min, pi_buy_max = 0.10, 0.50
    pi_rev_min, pi_rev_max = 0.05, 0.30
    weights = {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}

    # EVs (list of dicts) using Table I parameters. Each EV has id, Ecap (kWh), SoC_init, SoC_max, R_i (kW), T_stay (h)
    ev_list = [
        {'id': 1, 'Ecap': 60.0, 'SoC_init': 0.30, 'SoC_max': 0.80, 'R_i': 7.0, 'T_stay': 4.0},
        {'id': 2, 'Ecap': 40.0, 'SoC_init': 0.60, 'SoC_max': 0.90, 'R_i': 11.0, 'T_stay': 1.0},
        {'id': 3, 'Ecap': 50.0, 'SoC_init': 0.20, 'SoC_max': 0.60, 'R_i': 7.0, 'T_stay': 6.0},
    ]

    # Compute all factors and priority scores
    lambda_scores, details = calc_priority_scores(ev_list, weights, cdeg, P_avg, P_max,
                                                 pi_buy, pi_rev,
                                                 pi_buy_min, pi_buy_max,
                                                 pi_rev_min, pi_rev_max)

    # Print intermediate values (match formatting used in paper)
    print("=== Stage-I Priority Scheduling (Numerical Example) ===\n")
    print("Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):")
    for ev in ev_list:
        dE = details[ev['id']]['DeltaE']
        p_req = details[ev['id']]['p_req']
        phi = details[ev['id']]['phi']
        # print with rounding similar to paper
        print(f"EV{ev['id']}: DeltaE = {dE:.3f} kWh, p_req = {p_req:.3f} kW, phi = {phi:.3f}")
    print()

    print("Step 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):")
    for ev in ev_list:
        raw_deg = cdeg * details[ev['id']]['DeltaE']
        D = details[ev['id']]['Dfactor']
        print(f"EV{ev['id']}: cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {D:.3f}")
    print()

    # Grid stress and price (system-level)
    G_factor = next(iter(details.values()))['Gfactor']  # same for all
    P_factor = next(iter(details.values()))['Pfactor']
    print("Step 3 — Grid stress factor (Gfactor) (Eq. (24)):")
    print(f"Gfactor = {G_factor:.4f}")
    print()
    print("Step 4 — Price factor (Pfactor) (Eq. (25),(26)):")
    print(f"Pfactor = {P_factor:.3f}")
    print()

    # Lambda values
    print("Step 5 — Priority scores (lambda_i) (Eq. (22)):")
    for ev in ev_list:
        lam = details[ev['id']]['lambda']
        print(f"lambda_{ev['id']} = {lam:.5f}")
    print()

    # Sorting & assignment
    sorted_by_lambda = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    print("Step 6 — Ranking and Admission:")
    rank_str = " > ".join([f"EV{eid}" for eid, _ in sorted_by_lambda])
    print(f"Ranking (desc lambda): {rank_str}")
    admitted = assign_chargers(lambda_scores, M)
    print(f"With M={M} chargers, admitted EVs: {', '.join('EV'+str(x) for x in admitted)}")
    waiting = [ev['id'] for ev in ev_list if ev['id'] not in admitted]
    print(f"Waiting EVs: {', '.join('EV'+str(x) for x in waiting)}")

if __name__ == "__main__":
    main()
