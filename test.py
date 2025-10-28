#!/usr/bin/env python3
"""
pipeline_full_normalized_general.py

Integrated Stage-I (Priority Scheduling) and Stage-II (GA) pipeline.
- Loads EV definitions from CSV or JSON (required fields: id,Ecap,SoC_init,SoC_max,R_i,T_stay).
- Stage-I: priority scheduling (select top-M EVs).
- Stage-II: GA-based power scheduling over admitted EVs (uses DEAP).
- CLI:
    --ev-file   : path to EV CSV/JSON (auto-detects 'evs_30.csv' if present)
    --chargers  : number of chargers (M). Default = 10
"""

import argparse
import random
import math
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# DEAP imports
from deap import base, creator, tools

# -------------------------
# Utility: load EV pool from file (CSV or JSON)
# -------------------------
def load_ev_pool(file_path: str) -> List[Dict]:
    """
    Load EV definitions from CSV or JSON file.
    Required fields: id, Ecap, SoC_init, SoC_max, R_i, T_stay
    Optional fields: T_arr_idx, T_dep_idx, SoC_min, cdeg
    Returns list of EV dicts.
    """
    if file_path is None:
        raise ValueError("No file_path provided")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
        records = df.to_dict(orient='records')
    elif ext in ('.json', '.jsn'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            list_candidates = [v for v in data.values() if isinstance(v, list)]
            records = list_candidates[0]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("JSON must be a list of EV objects or contain a list value.")
    else:
        raise ValueError("Unsupported file extension. Use .csv or .json")

    evs = []
    required = {'id', 'Ecap', 'SoC_init', 'SoC_max', 'R_i', 'T_stay'}
    for rec in records:
        if not required.issubset(set(rec.keys())):
            missing = required - set(rec.keys())
            raise ValueError(f"Missing required columns/keys in EV definition: {missing}")
        ev = {}
        ev['id'] = int(rec['id'])
        ev['Ecap'] = float(rec['Ecap'])
        ev['SoC_init'] = float(rec['SoC_init'])
        ev['SoC_max'] = float(rec['SoC_max'])
        ev['R_i'] = float(rec['R_i'])
        ev['T_stay'] = float(rec['T_stay'])
        if 'T_arr_idx' in rec and not pd.isna(rec['T_arr_idx']):
            ev['T_arr_idx'] = int(rec['T_arr_idx'])
        if 'T_dep_idx' in rec and not pd.isna(rec['T_dep_idx']):
            ev['T_dep_idx'] = int(rec['T_dep_idx'])
        if 'SoC_min' in rec and not pd.isna(rec['SoC_min']):
            ev['SoC_min'] = float(rec['SoC_min'])
        else:
            ev['SoC_min'] = 0.0
        if 'cdeg' in rec and not pd.isna(rec['cdeg']):
            ev['cdeg'] = float(rec['cdeg'])
        else:
            ev['cdeg'] = 0.02
        evs.append(ev)
    return evs

# -------------------------
# Stage-I: Priority Scheduling
# -------------------------
def calc_delta_E(ev: Dict) -> float:
    """Eq. (7): DeltaE = (SoC_max - SoC_init) * Ecap (kWh)"""
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    """Eq. (9): p_req = DeltaE / T_stay, capped at R_i (paper's example uses this cap)."""
    deltaE = calc_delta_E(ev)
    if ev['T_stay'] <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / ev['T_stay']
    return min(raw, ev['R_i'])

def calc_phi(ev: Dict) -> float:
    """Eq. (10): urgency phi = min(1, p_req / P_ref)"""
    p_req = calc_p_req(ev)
    pref = ev['R_i']
    if pref <= 0:
        return 0.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], default_cdeg: float) -> Dict[int, float]:
    """
    Eq. (23) min-max normalization of cdeg_i * DeltaE_i.
    Uses per-EV cdeg if present, otherwise default_cdeg.
    """
    values = []
    for ev in ev_list:
        cdeg_i = ev.get('cdeg', default_cdeg)
        values.append(cdeg_i * calc_delta_E(ev))
    vmin, vmax = min(values), max(values)
    denom = vmax - vmin if vmax != vmin else 1.0
    return {ev['id']: (ev.get('cdeg', default_cdeg) * calc_delta_E(ev) - vmin) / denom for ev in ev_list}

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    """
    Eq. (24): Gfactor = max(0, (P_hat_agg - P_avg)/(P_max - P_avg))
    Capped to 1.0 for stability when many EVs exist.
    """
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = P_max - P_avg if P_max != P_avg else 1.0
    raw = max(0.0, (p_hat - P_avg) / denom)
    return min(raw, 1.0)

def _mean_if_array(x):
    if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        return float(np.mean(x))
    return float(x)

def calc_price_factor(pi_buy, pi_rev,
                      pi_buy_min: float, pi_buy_max: float,
                      pi_rev_min: float, pi_rev_max: float) -> float:
    """
    Eq. (25)-(26): normalize prices; accepts scalar or arrays (uses mean for arrays).
    """
    pi_buy_val = _mean_if_array(pi_buy)
    pi_rev_val = _mean_if_array(pi_rev)
    buy_denom = (pi_buy_max - pi_buy_min) if (pi_buy_max - pi_buy_min) != 0 else 1.0
    rev_denom = (pi_rev_max - pi_rev_min) if (pi_rev_max - pi_rev_min) != 0 else 1.0
    P_buy = (pi_buy_val - pi_buy_min) / buy_denom
    P_rev = (pi_rev_val - pi_rev_min) / rev_denom
    return P_buy - P_rev

def calc_priority_scores(ev_list: List[Dict], weights: Dict[str, float],
                         default_cdeg: float, P_avg: float, P_max: float,
                         pi_buy, pi_rev,
                         pi_buy_min: float, pi_buy_max: float,
                         pi_rev_min: float, pi_rev_max: float) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    """
    Compute phi, Dfactor, Gfactor, Pfactor and lambda for each EV (Eq. (22)).
    Returns lambda_scores dict and details per EV.
    """
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}
    D_map = calc_degradation_factor(ev_list, default_cdeg)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)
    P_factor = calc_price_factor(pi_buy, pi_rev, pi_buy_min, pi_buy_max, pi_rev_min, pi_rev_max)

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
            'DeltaE': calc_delta_E(ev),
            'cdeg_used': ev.get('cdeg', default_cdeg)
        }
    return lambda_scores, details

def assign_chargers(lambda_scores: Dict[int, float], M: int) -> List[int]:
    """Rank EVs by descending lambda; tie-breaker: lower EV id first. Return top-M IDs."""
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    return [eid for eid, _ in sorted_evs[:M]]

def run_stage1(ev_list: List[Dict], system_params: Dict) -> Tuple[List[Dict], Dict]:
    """Run Stage-I and print details (matches earlier outputs)."""
    lambda_scores, details = calc_priority_scores(
        ev_list,
        system_params['weights'],
        system_params['cdeg'],
        system_params['P_avg'],
        system_params['P_max'],
        system_params['pi_buy'],
        system_params['pi_rev'],
        system_params['pi_buy_min'],
        system_params['pi_buy_max'],
        system_params['pi_rev_min'],
        system_params['pi_rev_max']
    )
    admitted_ids = assign_chargers(lambda_scores, system_params['M'])
    admitted = [ev for ev in ev_list if ev['id'] in admitted_ids]

    # Print summary
    print("=== Stage-I Priority Scheduling (Numerical Example) ===\n")
    print("Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):")
    for ev in ev_list:
        dE = details[ev['id']]['DeltaE']
        p_req = details[ev['id']]['p_req']
        phi = details[ev['id']]['phi']
        print(f"EV{ev['id']}: DeltaE = {dE:.3f} kWh, p_req = {p_req:.3f} kW, phi = {phi:.3f}")
    print()
    print("Step 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):")
    for ev in ev_list:
        cdeg_used = details[ev['id']]['cdeg_used']
        raw_deg = cdeg_used * details[ev['id']]['DeltaE']
        D = details[ev['id']]['Dfactor']
        print(f"EV{ev['id']}: cdeg_used = {cdeg_used:.4f}, cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {D:.3f}")
    print()
    G_factor = next(iter(details.values()))['Gfactor']
    P_factor = next(iter(details.values()))['Pfactor']
    print("Step 3 — Grid stress factor (Gfactor) (Eq. (24)):")
    print(f"Gfactor (capped <=1.0) = {G_factor:.4f}")
    print()
    print("Step 4 — Price factor (Pfactor) (Eq. (25),(26)):")
    print(f"Pfactor = {P_factor:.3f}\n")
    print("Step 5 — Priority scores (lambda_i) (Eq. (22)):")
    for ev in ev_list:
        lam = details[ev['id']]['lambda']
        print(f"lambda_{ev['id']} = {lam:.5f}")
    print()
    sorted_by_lambda = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    print("Step 6 — Ranking and Admission:")
    rank_str = " > ".join([f"EV{eid}" for eid, _ in sorted_by_lambda])
    print(f"Ranking (desc lambda): {rank_str}")
    print(f"With M={system_params['M']} chargers, admitted EVs: {', '.join('EV'+str(x) for x in admitted_ids)}")
    waiting = [ev['id'] for ev in ev_list if ev['id'] not in admitted_ids]
    print(f"Waiting EVs: {', '.join('EV'+str(x) for x in waiting)}")
    return admitted, details

# -------------------------
# Stage-II: GA-based power scheduling (normalized + elitism)
# -------------------------
def flatten_index(i, t, T): return i * T + t
def unflatten(individual, M, T): return np.array(individual, dtype=float).reshape((M, T))

def compute_F1(net_schedule, pi_buy, pi_rev, delta_t):
    M, T = net_schedule.shape
    cost = 0.0
    for t in range(T):
        for i in range(M):
            p = net_schedule[i, t]
            if p > 0:
                cost += pi_buy[t] * p * delta_t
            elif p < 0:
                cost += pi_rev[t] * p * delta_t
    return cost

def compute_F2(net_schedule, cdeg_arr, delta_t):
    M, T = net_schedule.shape
    return sum(cdeg_arr[i] * abs(net_schedule[i, t]) * delta_t for i in range(M) for t in range(T))

def compute_F3(net_schedule):
    L = np.sum(net_schedule, axis=0)
    Lbar = np.mean(L)
    return np.sum((L - Lbar) ** 2)

def compute_Si_from_schedule(net_schedule, evs, delta_t):
    M, T = net_schedule.shape
    Si_list = [0.0] * M
    for idx in range(M):
        ev = evs[idx]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        SoC_max = ev['SoC_max']
        for t in range(T):
            p = net_schedule[idx, t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
        SoC_T = SoC
        Ereq = max(0.0, (SoC_max - ev['SoC_init']) * Ecap)
        Tstay = ev['T_stay']
        preq = (Ereq / Tstay) if Tstay > 0 else (float('inf') if Ereq > 0 else 0.0)
        phi_i = min(1.0, preq / ev['P_ref']) if ev['P_ref'] > 0 else 0.0
        if SoC_T >= SoC_max:
            delta_i = 0.0
        else:
            denom = (SoC_max - ev['SoC_init'])
            delta_i = (SoC_max - SoC_T) / denom if denom > 0 else 1.0
            if delta_i < 0.0:
                delta_i = 0.0
        Si_list[idx] = max(0.0, 1.0 - phi_i * delta_i)
    return Si_list

def compute_F4(net_schedule, evs, delta_t):
    Si = compute_Si_from_schedule(net_schedule, evs, delta_t)
    return sum(Si), Si

def compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t):
    M, T = net_schedule.shape
    V_SoC = 0.0
    eps = 1e-9
    for i in range(M):
        ev = evs[i]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        SoC_min = ev.get('SoC_min', 0.0)
        SoC_max = ev['SoC_max']
        for t in range(T):
            p = net_schedule[i, t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
            if SoC < SoC_min - eps:
                V_SoC += (SoC_min - SoC)
            elif SoC > SoC_max + eps:
                V_SoC += (SoC - SoC_max)
    V_grid = 0.0
    V_occ = 0.0
    for t in range(T):
        Lt = float(np.sum(net_schedule[:, t]))
        if Lt > P_max:
            V_grid += (Lt - P_max)
        active = int(np.sum(np.abs(net_schedule[:, t]) > 1e-6))
        if active > num_chargers:
            V_occ += (active - num_chargers)
    return V_SoC, V_occ, V_grid

def make_fitness_function(evs, M, T, delta_t,
                          pi_buy, pi_rev,
                          cdeg_arr,
                          P_max, num_chargers,
                          w1, w2, w3, w4,
                          alpha1, alpha2, alpha3):
    # simple normalization denominators
    max_power_sum = sum(ev['P_ref'] for ev in evs) if evs else 1.0
    max_energy_over_horizon = max_power_sum * delta_t * T
    max_price = max(max(pi_buy) if hasattr(pi_buy, '__len__') else pi_buy,
                    max(pi_rev) if hasattr(pi_rev, '__len__') else pi_rev)
    max_cdeg = max(cdeg_arr) if len(cdeg_arr) > 0 else 1.0
    denom_F1 = (max_price * max_energy_over_horizon + 1e-9)
    denom_F2 = (max_cdeg * max_energy_over_horizon + 1e-9)
    denom_F3 = ((max_power_sum ** 2) * T + 1e-9)
    denom_F4 = (M + 1e-9)
    denom_Omega = (max_power_sum * delta_t * T + 1e-9)

    def evaluate(individual):
        net_schedule = unflatten(individual, M, T)
        F1 = compute_F1(net_schedule, pi_buy, pi_rev, delta_t)
        F2 = compute_F2(net_schedule, cdeg_arr, delta_t)
        F3 = compute_F3(net_schedule)
        F4_sum, Si_list = compute_F4(net_schedule, evs, delta_t)
        V_SoC, V_occ, V_grid = compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t)
        Omega_raw = alpha1 * V_SoC + alpha2 * V_occ + alpha3 * V_grid
        F1_norm = F1 / denom_F1
        F2_norm = F2 / denom_F2
        F3_norm = F3 / denom_F3
        F4_norm = F4_sum / denom_F4
        Omega_norm = Omega_raw / denom_Omega
        J = w1 * F1_norm + w2 * F2_norm + w3 * F3_norm - w4 * F4_norm + Omega_norm
        individual._cached = {
            'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4_sum,
            'F1_norm': F1_norm, 'F2_norm': F2_norm, 'F3_norm': F3_norm, 'F4_norm': F4_norm,
            'Si_list': Si_list,
            'V_SoC': V_SoC, 'V_occ': V_occ, 'V_grid': V_grid,
            'Omega_raw': Omega_raw, 'Omega_norm': Omega_norm,
            'J': J
        }
        return (J,)
    return evaluate

def build_bounds_and_mask(evs, M, T):
    lower = [0.0] * (M * T)
    upper = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        pmin = -ev['P_ref']
        pmax = ev['P_ref']
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(T):
            idx = flatten_index(i, t, T)
            if Tarr <= t < Tdep:
                lower[idx] = pmin
                upper[idx] = pmax
            else:
                lower[idx] = 0.0
                upper[idx] = 0.0
    return lower, upper

def seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25):
    chrom = [0.0] * (M * T)
    price_arr = np.array(pi_buy)
    for i in range(M):
        ev = evs[i]
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        Tstay = ev['T_stay'] if ev['T_stay'] > 0 else 1e-9
        raw_preq = DeltaE / Tstay
        preq = min(raw_preq, ev['P_ref'])
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            continue
        stay_slots = list(range(Tarr, Tdep))
        prices_in_stay = price_arr[stay_slots]
        k = max(1, int(math.ceil(seed_fraction * len(stay_slots))))
        chosen_idx_rel = list(np.argsort(prices_in_stay)[:k])
        for rel in chosen_idx_rel:
            t = stay_slots[rel]
            chrom[flatten_index(i, t, T)] = preq
    return chrom

def run_ga(evs, admitted_ids,
           T, delta_t,
           pi_buy, pi_rev,
           P_max,
           weights,
           alpha1, alpha2, alpha3,
           pop_size=100, ngen=300,
           cxpb=0.9, mutpb=0.2, eta_c=20.0, eta_m=20.0,
           tournament_size=3,
           stagnation_generations=40,
           seed_count=10,
           elitism_k=2,
           verbose=True):
    random.seed(42)
    M = len(evs)
    cdeg_arr = [ev.get('cdeg', 0.02) for ev in evs]
    num_chargers = len(evs)
    w1 = weights.get('w1', 0.25)
    w2 = weights.get('w2', 0.25)
    w3 = weights.get('w3', 0.25)
    w4 = weights.get('w4', 0.25)

    lower, upper = build_bounds_and_mask(evs, M, T)
    lower_arr = lower
    upper_arr = upper

    try:
        creator.FitnessMin
    except Exception:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def attr_float_at_index(idx):
        lo = lower_arr[idx]
        up = upper_arr[idx]
        if lo == up:
            return lo
        return random.uniform(lo, up)

    def generate_individual():
        indiv = [0.0] * (M * T)
        for idx in range(M * T):
            indiv[idx] = attr_float_at_index(idx)
        return creator.Individual(indiv)

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_fn = make_fitness_function(evs, M, T, delta_t,
                                    pi_buy, pi_rev,
                                    cdeg_arr,
                                    P_max, num_chargers,
                                    w1, w2, w3, w4,
                                    alpha1, alpha2, alpha3)
    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    _EPS = 1e-9
    safe_low = list(lower_arr)
    safe_up = list(upper_arr)
    fixed_mask = [False] * (M * T)
    for idx in range(M * T):
        if abs(lower_arr[idx] - upper_arr[idx]) < 1e-12:
            fixed_mask[idx] = True
            safe_up[idx] = safe_low[idx] + _EPS

    def safe_mate(ind1, ind2):
        tools.cxSimulatedBinaryBounded(ind1, ind2, low=safe_low, up=safe_up, eta=eta_c)
        for idx in range(M * T):
            if fixed_mask[idx]:
                ind1[idx] = lower_arr[idx]
                ind2[idx] = lower_arr[idx]
        return (ind1, ind2)

    def safe_mutate(individual):
        tools.mutPolynomialBounded(individual, low=safe_low, up=safe_up, eta=eta_m, indpb=1.0/(M*T))
        for idx in range(M * T):
            if fixed_mask[idx]:
                individual[idx] = lower_arr[idx]
        return (individual,)

    toolbox.register("mate", safe_mate)
    toolbox.register("mutate", safe_mutate)

    pop = toolbox.population(n=pop_size)
    num_seed = min(seed_count, pop_size)
    for s in range(num_seed):
        seed_chrom = seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25)
        pop[s][:] = seed_chrom

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best = tools.selBest(pop, 1)[0]
    best_J = best.fitness.values[0]
    no_improve = 0
    gen = 0
    if verbose:
        print(f"GA start: pop_size={pop_size}, ngen={ngen}, M={M}, T={T}")
        print("Initial best J =", round(best_J, 8))

    while gen < ngen and no_improve < stagnation_generations:
        gen += 1
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            if random.random() <= cxpb:
                toolbox.mate(offspring[i], offspring[i + 1])

        # mutation
        for i in range(len(offspring)):
            if random.random() <= mutpb:
                toolbox.mutate(offspring[i])

        # evaluate new individuals
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # elitism: keep top-k from current pop
        elites = tools.selBest(pop, elitism_k)
        combined = offspring + list(map(toolbox.clone, elites))
        for ind in combined:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        combined.sort(key=lambda ind: ind.fitness.values[0])
        pop = combined[:pop_size]

        current_best = tools.selBest(pop, 1)[0]
        current_best_J = current_best.fitness.values[0]
        if current_best_J + 1e-12 < best_J:
            best = toolbox.clone(current_best)
            best_J = current_best_J
            no_improve = 0
            if verbose:
                print(f"Gen {gen} improved best J -> {best_J:.8f}")
        else:
            no_improve += 1
            if verbose and gen % 10 == 0:
                print(f"Gen {gen} best J {best_J:.8f} (no improve {no_improve})")

    best_schedule = unflatten(best, M, T)
    breakdown = getattr(best, "_cached", None)
    if breakdown is None:
        best.fitness.values = toolbox.evaluate(best)
        breakdown = best._cached

    result = {
        'best_schedule': best_schedule,
        'J': breakdown['J'],
        'F1': breakdown['F1'], 'F2': breakdown['F2'], 'F3': breakdown['F3'], 'F4': breakdown['F4'],
        'F1_norm': breakdown.get('F1_norm'), 'F2_norm': breakdown.get('F2_norm'),
        'F3_norm': breakdown.get('F3_norm'), 'F4_norm': breakdown.get('F4_norm'),
        'Si_list': breakdown['Si_list'],
        'V_SoC': breakdown['V_SoC'], 'V_occ': breakdown['V_occ'], 'V_grid': breakdown['V_grid'],
        'Omega_raw': breakdown['Omega_raw'], 'Omega_norm': breakdown['Omega_norm'],
        'generations_executed': gen
    }
    return result

# -------------------------
# Orchestrator main()
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Stage-I priority scheduling and Stage-II GA power scheduling.")
    parser.add_argument('--ev-file', type=str, default=None,
                        help='Path to EV definitions file (CSV or JSON). If omitted, will auto-detect evs_30.csv or use a small built-in example.')
    parser.add_argument('--chargers', type=int, default=10,
                        help='Number of chargers (M). Default = 10')
    args = parser.parse_args()

    # Auto-detect evs_30.csv when no --ev-file specified
    if args.ev_file is None and os.path.exists('evs_30.csv'):
        args.ev_file = 'evs_30.csv'
        print("No --ev-file provided; found 'evs_30.csv' in current directory and will use it.")

    # Load EV pool (either from file or default example)
    if args.ev_file:
        try:
            ev_pool = load_ev_pool(args.ev_file)
            print(f"Loaded {len(ev_pool)} EVs from {args.ev_file}")
        except Exception as e:
            print("Error loading EV file:", e)
            return
    else:
        ev_pool = [
            {'id': 1, 'Ecap': 60.0, 'SoC_init': 0.30, 'SoC_max': 0.80, 'R_i': 7.0, 'T_stay': 4.0},
            {'id': 2, 'Ecap': 40.0, 'SoC_init': 0.60, 'SoC_max': 0.90, 'R_i': 11.0, 'T_stay': 1.0},
            {'id': 3, 'Ecap': 50.0, 'SoC_init': 0.20, 'SoC_max': 0.60, 'R_i': 7.0, 'T_stay': 6.0},
        ]
        print("Using default 3-EV example (no --ev-file provided).")

    # System parameters
    system_params = {
        'M': args.chargers,    # <-- number of chargers requested by user
        'P_max': 25.0,
        'P_avg': 12.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }
    print(f"Number of chargers M set to: {system_params['M']}")

    # Stage-I
    admitted, details = run_stage1(ev_pool, system_params)
    print("\nStage-I admitted EV ids:", [ev['id'] for ev in admitted])

    # Convert admitted EVs for Stage-II: map R_i -> P_ref, set T_arr_idx/T_dep_idx if absent
    T = 48
    delta_t = 0.25
    admitted_evs = []
    for ev in admitted:
        Tarr = ev.get('T_arr_idx', None)
        Tdep = ev.get('T_dep_idx', None)
        if Tarr is None or Tdep is None:
            slots = max(1, int(math.ceil(ev['T_stay'] / delta_t)))
            Tarr = 0
            Tdep = min(T, slots)
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'P_ref': ev['R_i'],
            'T_stay': ev['T_stay'],
            'T_arr_idx': Tarr,
            'T_dep_idx': Tdep,
            'cdeg': ev.get('cdeg', 0.02),
            'SoC_min': ev.get('SoC_min', 0.0)
        })

    # GA inputs and hyperparams
    pi_buy_arr = [0.25] * T
    pi_rev_arr = [0.18] * T
    P_max = 25.0
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    alpha1, alpha2, alpha3 = 50.0, 50.0, 50.0

    # Run Stage-II GA
    result = run_ga(evs=admitted_evs, admitted_ids=None,
                    T=T, delta_t=delta_t,
                    pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
                    P_max=P_max,
                    weights=weights,
                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                    pop_size=120, ngen=300,
                    cxpb=0.9, mutpb=0.3,
                    eta_c=20.0, eta_m=20.0,
                    tournament_size=3,
                    stagnation_generations=60,
                    seed_count=20,
                    elitism_k=4,
                    verbose=True)

    # Print and save GA outputs
    print("\n=== GA Result Summary ===")
    print("Objective J (normalized weighted):", result['J'])
    print("F1 (cost, raw) :", result['F1'], "F1_norm:", result['F1_norm'])
    print("F2 (degradation, raw):", result['F2'], "F2_norm:", result['F2_norm'])
    print("F3 (variance, raw):", result['F3'], "F3_norm:", result['F3_norm'])
    print("F4 (satisfaction, raw):", result['F4'], "F4_norm:", result['F4_norm'])
    print("Penalties (raw):", result['Omega_raw'], "Penalties_norm:", result['Omega_norm'])
    print("Violations (SoC, occ, grid):", result['V_SoC'], result['V_occ'], result['V_grid'])
    print("Generations executed:", result['generations_executed'])

    best = result['best_schedule']
    df = pd.DataFrame(best, index=[ev['id'] for ev in admitted_evs])
    csv_path = os.path.abspath('best_schedule_normalized.csv')
    df.to_csv(csv_path, index=True)
    print(f"Saved best_schedule_normalized.csv to: {csv_path}")

if __name__ == "__main__":
    main()
