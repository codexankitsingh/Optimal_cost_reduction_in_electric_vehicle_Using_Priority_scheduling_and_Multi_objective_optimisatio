#!/usr/bin/env python3
"""
pipeline_updated.py

Full updated pipeline:
- Stage-I priority scheduling (general n EVs from CSV/JSON)
- Stage-II GA (DEAP) with normalization, seeding, repair, improved bounds
- CLI options: --ev-file, --chargers, with sensible defaults
"""

import argparse
import csv
import json
import math
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

# -------------------------
# Utility: load EVs (CSV or JSON)
# -------------------------
def load_ev_pool(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    evs = []
    if ext in ('.csv',):
        with open(path, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # convert numeric fields robustly; fall back to sensible defaults
                def f(key, default=None, typ=float):
                    v = row.get(key, None)
                    if v is None or v == '':
                        return default
                    try:
                        return typ(v)
                    except:
                        return default

                ev = {
                    'id': int(f('id', default=0, typ=int)),
                    'Ecap': f('Ecap', default=40.0),
                    'SoC_init': f('SoC_init', default=0.2),
                    'SoC_max': f('SoC_max', default=0.8),
                    'R_i': f('R_i', default=None),
                    'P_ref': f('P_ref', default=None),
                    'T_stay': f('T_stay', default=4.0),
                    'T_arr_idx': int(f('T_arr_idx', default=0, typ=int)),
                    'cdeg': f('cdeg', default=0.02),
                }
                evs.append(ev)
    elif ext in ('.json', '.jsn'):
        with open(path, 'r') as fh:
            data = json.load(fh)
            for row in data:
                ev = {
                    'id': int(row.get('id', 0)),
                    'Ecap': float(row.get('Ecap', 40.0)),
                    'SoC_init': float(row.get('SoC_init', 0.2)),
                    'SoC_max': float(row.get('SoC_max', 0.8)),
                    'R_i': row.get('R_i', None),
                    'P_ref': row.get('P_ref', None),
                    'T_stay': float(row.get('T_stay', 4.0)),
                    'T_arr_idx': int(row.get('T_arr_idx', 0)),
                    'cdeg': float(row.get('cdeg', 0.02)),
                }
                evs.append(ev)
    else:
        raise ValueError("Unsupported EV file type: must be CSV or JSON")
    return evs


# -------------------------
# Stage-I helpers
# -------------------------
def calc_delta_E(ev: Dict) -> float:
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    deltaE = calc_delta_E(ev)
    T_stay = ev.get('T_stay', 0.0)
    if T_stay <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / T_stay
    R_i = ev.get('R_i', ev.get('P_ref', None))
    if R_i is None or R_i <= 0:
        # fallback to a realistic default (7 kW)
        R_i = 7.0
    return min(raw, R_i)

def calc_phi(ev: Dict) -> float:
    p_req = calc_p_req(ev)
    pref = ev.get('R_i', ev.get('P_ref', None))
    if pref is None or pref <= 0:
        pref = 7.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], cdeg: float) -> Dict[int, float]:
    vals = [ (ev.get('cdeg', cdeg) * calc_delta_E(ev)) for ev in ev_list ]
    vmin, vmax = min(vals), max(vals)
    denom = (vmax - vmin) if vmax != vmin else 1.0
    return { ev['id']: (ev.get('cdeg', cdeg) * calc_delta_E(ev) - vmin) / denom for ev in ev_list }

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = (P_max - P_avg) if (P_max - P_avg) != 0 else 1.0
    return max(0.0, (p_hat - P_avg)/denom)

def calc_price_factor(pi_buy: float, pi_rev: float,
                      pi_buy_min: float, pi_buy_max: float,
                      pi_rev_min: float, pi_rev_max: float) -> float:
    buy_denom = (pi_buy_max - pi_buy_min) if (pi_buy_max - pi_buy_min) != 0 else 1.0
    rev_denom = (pi_rev_max - pi_rev_min) if (pi_rev_max - pi_rev_min) != 0 else 1.0
    P_buy = (pi_buy - pi_buy_min) / buy_denom
    P_rev = (pi_rev - pi_rev_min) / rev_denom
    return P_buy - P_rev

def calc_priority_scores(ev_list: List[Dict], weights: Dict[str, float],
                         cdeg: float, P_avg: float, P_max: float,
                         pi_buy: float, pi_rev: float,
                         pi_buy_min: float, pi_buy_max: float,
                         pi_rev_min: float, pi_rev_max: float):
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}
    D_map = calc_degradation_factor(ev_list, cdeg)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)
    P_factor = calc_price_factor(pi_buy, pi_rev, pi_buy_min, pi_buy_max, pi_rev_min, pi_rev_max)
    lambda_scores = {}
    details = {}
    for ev in ev_list:
        eid = ev['id']
        phi = phi_map[eid]
        D = D_map[eid]
        lam = (weights['w_s'] * phi - weights['w_d'] * D - weights['w_g'] * G_factor - weights['w_p'] * P_factor)
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
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    return [eid for eid,_ in sorted_evs[:M]]

def run_stage1(ev_list: List[Dict], system_params: Dict) -> Tuple[List[Dict], Dict]:
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

    print("=== Stage-I Priority Scheduling ===\n")
    print("Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):")
    for ev in ev_list:
        d = details[ev['id']]
        print(f"EV{ev['id']}: DeltaE = {d['DeltaE']:.3f} kWh, p_req = {d['p_req']:.3f} kW, phi = {d['phi']:.3f}")
    print("\nStep 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):")
    for ev in ev_list:
        raw_deg = system_params['cdeg'] * details[ev['id']]['DeltaE']
        print(f"EV{ev['id']}: cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {details[ev['id']]['Dfactor']:.3f}")
    print(f"\nStep 3 — Grid stress factor (Gfactor) (Eq. (24)):\nGfactor = {next(iter(details.values()))['Gfactor']:.4f}")
    print(f"\nStep 4 — Price factor (Pfactor) (Eq. (25),(26)):\nPfactor = {next(iter(details.values()))['Pfactor']:.3f}\n")
    print("Step 5 — Priority scores (lambda_i) (Eq. (22)):")
    for ev in ev_list:
        lam = details[ev['id']]['lambda']
        print(f"lambda_{ev['id']} = {lam:.5f}")
    print()
    sorted_by_lambda = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    rank_str = " > ".join([f"EV{eid}" for eid,_ in sorted_by_lambda])
    print("Step 6 — Ranking and Admission:")
    print(f"Ranking (desc lambda): {rank_str}")
    print(f"With M={system_params['M']} chargers, admitted EVs: {', '.join('EV'+str(x) for x in admitted_ids)}")
    waiting = [ev['id'] for ev in ev_list if ev['id'] not in admitted_ids]
    print(f"Waiting EVs: {', '.join('EV'+str(x) for x in waiting)}\n")
    return admitted, details

# -------------------------
# Stage-II (GA) helpers
# -------------------------
def flatten_index(i, t, T): return i * T + t
def unflatten(individual, M, T): return np.array(individual, dtype=float).reshape((M, T))

def compute_F1(net_schedule, pi_buy, pi_rev, delta_t):
    M, T = net_schedule.shape
    cost = 0.0
    for t in range(T):
        for i in range(M):
            p = net_schedule[i,t]
            if p > 0:
                cost += pi_buy[t] * p * delta_t
            elif p < 0:
                cost += pi_rev[t] * p * delta_t
    return cost

def compute_F2(net_schedule, cdeg_arr, delta_t):
    M, T = net_schedule.shape
    return sum(cdeg_arr[i] * abs(net_schedule[i,t]) * delta_t for i in range(M) for t in range(T))

def compute_F3(net_schedule):
    L = np.sum(net_schedule, axis=0)
    Lbar = np.mean(L)
    return np.sum((L - Lbar)**2)

def compute_Si_from_schedule(net_schedule, evs, delta_t):
    M, T = net_schedule.shape
    Si_list = [0.0] * M
    for idx in range(M):
        ev = evs[idx]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        SoC_max = ev['SoC_max']
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(Tarr, Tdep):
            p = net_schedule[idx, t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
        SoC_T = SoC
        Ereq = max(0.0, (SoC_max - ev['SoC_init']) * Ecap)
        Tstay = max(ev.get('T_stay', 1e-9), 1e-9)
        preq = (Ereq / Tstay) if Tstay > 0 else (float('inf') if Ereq > 0 else 0.0)
        pref = ev.get('P_ref', ev.get('R_i', 7.0))
        if pref is None or pref <= 0:
            pref = 7.0
        phi_i = min(1.0, preq / pref)
        if SoC_T >= SoC_max:
            delta_i = 0.0
        else:
            denom = (SoC_max - ev['SoC_init'])
            delta_i = (SoC_max - SoC_T) / denom if denom > 0 else 1.0
            delta_i = max(0.0, delta_i)
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
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(Tarr, Tdep):
            p = net_schedule[i,t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
            if SoC < SoC_min - eps:
                V_SoC += (SoC_min - SoC)
            elif SoC > SoC_max + eps:
                V_SoC += (SoC - SoC_max)
    V_grid = 0.0
    V_occ = 0.0
    for t in range(T):
        Lt = float(np.sum(net_schedule[:,t]))
        if Lt > P_max:
            V_grid += (Lt - P_max)
        active = int(np.sum(np.abs(net_schedule[:,t]) > 1e-6))
        if active > num_chargers:
            V_occ += (active - num_chargers)
    return V_SoC, V_occ, V_grid

# -------------------------
# GA: fitness factory (with normalization)
# -------------------------
def make_fitness_function(evs, M, T, delta_t,
                          pi_buy, pi_rev,
                          cdeg_arr,
                          P_max, num_chargers,
                          w1, w2, w3, w4,
                          alpha1, alpha2, alpha3):
    max_power_sum = sum(ev.get('P_ref', ev.get('R_i', 7.0)) for ev in evs) if evs else 1.0
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

# -------------------------
# Bounds, seeding, repair
# -------------------------
def build_bounds_and_mask(evs, M, T):
    lower = [0.0] * (M * T)
    upper = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        pmin = 0.0  # only charging positive power in our model
        pmax = ev.get('P_ref', ev.get('R_i', 7.0))
        if pmax is None or pmax <= 0:
            pmax = 7.0
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
        Tstay = max(ev.get('T_stay', 0.0), 1e-9)
        raw_preq = DeltaE / Tstay
        preq = min(raw_preq, ev.get('P_ref', ev.get('R_i', 7.0)))
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            continue
        stay_slots = list(range(Tarr, Tdep))
        prices_in_stay = price_arr[stay_slots]
        k = max(1, int(math.ceil(seed_fraction * len(stay_slots))))
        chosen_idx_rel = list(np.argsort(prices_in_stay)[:k])
        # distribute energy evenly across chosen slots
        for rel in chosen_idx_rel:
            t = stay_slots[rel]
            chrom[flatten_index(i, t, T)] = preq
    return chrom

def repair_individual(individual, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers):
    """Repair:
     - zero outside windows
     - scale/redistribute per-EV energy to try meet DeltaE (greedy to cheapest slots)
     - enforce per-slot active <= num_chargers by zeroing smallest contributors
    """
    mat = unflatten(individual, M, T)
    price = np.array(pi_buy)
    # Zero outside bounds and clamp
    for idx in range(M*T):
        if abs(upper_arr[idx] - lower_arr[idx]) < 1e-12:
            mat.flat[idx] = lower_arr[idx]
        else:
            mat.flat[idx] = max(lower_arr[idx], min(upper_arr[idx], mat.flat[idx]))

    # Per-EV energy repair
    for i in range(M):
        ev = evs[i]
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            mat[i, :] = 0.0
            continue
        slots = list(range(Tarr, Tdep))
        current_energy = float(np.sum(mat[i, slots]) * delta_t)
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        target_energy = min(DeltaE, ev.get('P_ref', ev.get('R_i', 7.0)) * len(slots) * delta_t)
        if abs(current_energy - target_energy) < 1e-6:
            continue
        # if current > target: scale down uniformly across active slots
        if current_energy > 0 and current_energy > target_energy:
            factor = target_energy / current_energy if current_energy > 0 else 0.0
            for t in slots:
                mat[i,t] *= factor
        elif current_energy < target_energy:
            # greedily fill cheapest slots (lowest price)
            need = target_energy - current_energy
            slot_order = sorted(slots, key=lambda t: price[t])
            for t in slot_order:
                avail = upper_arr[flatten_index(i, t, T)] - mat[i,t]
                if avail <= 1e-9:
                    continue
                add = min(avail * delta_t, need) / delta_t
                # add in kW such that energy added = add * delta_t
                mat[i,t] += add
                need -= add * delta_t
                if need <= 1e-9:
                    break
        # clamp per-slot
        for t in slots:
            mat[i,t] = max(lower_arr[flatten_index(i,t,T)], min(upper_arr[flatten_index(i,t,T)], mat[i,t]))

    # Enforce occupancy per slot: if active > num_chargers, zero smallest-power entries until ok
    for t in range(T):
        col = mat[:, t]
        active_idx = np.where(np.abs(col) > 1e-6)[0].tolist()
        if len(active_idx) <= num_chargers:
            continue
        # sort by power ascending (small contributors first)
        powers = [(idx, col[idx]) for idx in active_idx]
        powers_sorted = sorted(powers, key=lambda x: x[1])  # smallest first
        # remove smallest contributors until active <= num_chargers
        remove_count = len(active_idx) - num_chargers
        for j in range(remove_count):
            idx_to_zero = powers_sorted[j][0]
            mat[idx_to_zero, t] = 0.0

    # write back into individual
    flat = mat.flatten().tolist()
    individual[:] = flat
    return individual

# -------------------------
# GA orchestration
# -------------------------
def run_ga(evs, T, delta_t,
           pi_buy, pi_rev,
           P_max,
           weights,
           alpha1, alpha2, alpha3,
           pop_size=120, ngen=300,
           cxpb=0.9, mutpb=0.3, eta_c=20.0, eta_m=20.0,
           tournament_size=3,
           stagnation_generations=40,
           seed_count=10,
           elitism_k=2,
           num_chargers=10,
           verbose=True):
    random.seed(42)
    M = len(evs)
    if M == 0:
        raise ValueError("No admitted EVs passed to run_ga")
    cdeg_arr = [ev.get('cdeg', 0.02) for ev in evs]
    num_chargers = int(num_chargers)
    w1 = weights.get('w1', 0.25)
    w2 = weights.get('w2', 0.25)
    w3 = weights.get('w3', 0.25)
    w4 = weights.get('w4', 0.25)

    lower_arr, upper_arr = build_bounds_and_mask(evs, M, T)

    # DEAP creator
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
        lo = lower_arr[idx]; up = upper_arr[idx]
        if abs(up - lo) < 1e-12:
            return lo
        return random.uniform(lo, up)

    def generate_individual():
        indiv = [0.0] * (M * T)
        for idx in range(M * T):
            indiv[idx] = attr_float_at_index(idx)
        return creator.Individual(indiv)

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_fn = make_fitness_function(evs, M, T, delta_t, pi_buy, pi_rev, cdeg_arr, P_max, num_chargers,
                                    w1, w2, w3, w4, alpha1, alpha2, alpha3)
    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # safe bounded operators
    safe_low = list(lower_arr); safe_up = list(upper_arr)
    fixed_mask = [False] * (M*T)
    for idx in range(M*T):
        if abs(lower_arr[idx] - upper_arr[idx]) < 1e-12:
            fixed_mask[idx] = True
            safe_up[idx] = safe_low[idx] + 1e-9

    def safe_mate(ind1, ind2):
        tools.cxSimulatedBinaryBounded(ind1, ind2, low=safe_low, up=safe_up, eta=eta_c)
        for idx in range(M*T):
            if fixed_mask[idx]:
                ind1[idx] = lower_arr[idx]; ind2[idx] = lower_arr[idx]
        return ind1, ind2

    def safe_mutate(ind):
        tools.mutPolynomialBounded(ind, low=safe_low, up=safe_up, eta=eta_m, indpb=1.0/(M*T))
        for idx in range(M*T):
            if fixed_mask[idx]:
                ind[idx] = lower_arr[idx]
        return ind,

    toolbox.register("mate", safe_mate)
    toolbox.register("mutate", safe_mutate)

    pop = toolbox.population(n=pop_size)
    # seeding: use Stage-I seeds for first 'seed_count' individuals
    num_seed = min(seed_count, pop_size)
    for s in range(num_seed):
        seed_chrom = seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25 + 0.25*random.random())
        pop[s][:] = seed_chrom

    # evaluate initial population
    for ind in pop:
        # repair before evaluating
        repair_individual(ind, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers)
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
            if i+1 >= len(offspring): break
            if random.random() <= cxpb:
                toolbox.mate(offspring[i], offspring[i+1])
        # mutation
        for i in range(len(offspring)):
            if random.random() <= mutpb:
                toolbox.mutate(offspring[i])
        # repair & evaluate invalid
        for ind in offspring:
            repair_individual(ind, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers)
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit
        # elitism
        elites = tools.selBest(pop, elitism_k)
        combined = offspring + list(map(toolbox.clone, elites))
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

    # finalize
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
# Orchestrator main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev-file', type=str, default=None,
                        help='EV file (CSV or JSON). If omitted, looks for evs.csv / evs.json in cwd.')
    parser.add_argument('--chargers', type=int, default=10, help='Number of chargers (M). Default 10.')
    parser.add_argument('--T', type=int, default=48, help='Number of time slots (e.g. 48 for 24h half-hour slots).')
    parser.add_argument('--delta-t', type=float, default=0.5, help='Slot duration in hours (default 0.5).')
    args = parser.parse_args()

    # find default ev file if omitted
    ev_file = args.ev_file
    if ev_file is None:
        if os.path.exists('evs.csv'):
            ev_file = 'evs.csv'
            print("No --ev-file provided; found 'evs.csv' in current directory and will use it.")
        elif os.path.exists('evs.json'):
            ev_file = 'evs.json'
            print("No --ev-file provided; found 'evs.json' in current directory and will use it.")
        else:
            print("No EV file found. Exiting.")
            return

    ev_pool = load_ev_pool(ev_file)
    print(f"Loaded {len(ev_pool)} EVs from {ev_file}")

    # system params (tune if needed)
    system_params = {
        'M': args.chargers,
        'P_max': 100.0,
        'P_avg': 40.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }

    # Stage-I
    admitted, details = run_stage1(ev_pool, system_params)
    print("Stage-I admitted EV ids:", [ev['id'] for ev in admitted])

    # Stage-II: convert admitted EVs properly
    T = args.T
    delta_t = args.delta_t
    admitted_evs = []
    for ev in admitted:
        T_arr_idx = int(ev.get('T_arr_idx', 0))
        slots = max(1, int(math.ceil(ev.get('T_stay', 0.0) / delta_t)))
        T_dep_idx = min(T, T_arr_idx + slots)
        p_ref = ev.get('R_i', ev.get('P_ref', None))
        if p_ref is None or p_ref <= 0:
            p_ref = 7.0
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'P_ref': p_ref,
            'T_stay': ev['T_stay'],
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': ev.get('cdeg', system_params['cdeg'])
        })

    # GA hyperparams & inputs
    pi_buy_arr = [system_params['pi_buy']] * T
    pi_rev_arr = [system_params['pi_rev']] * T
    P_max = system_params['P_max']
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    alpha1, alpha2, alpha3 = 50.0, 50.0, 50.0

    # Run GA
    result = run_ga(evs=admitted_evs, T=T, delta_t=delta_t,
                    pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
                    P_max=P_max,
                    weights=weights,
                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                    pop_size=120, ngen=300,
                    cxpb=0.9, mutpb=0.3,
                    eta_c=20.0, eta_m=20.0,
                    tournament_size=3,
                    stagnation_generations=60,
                    seed_count=min(20, max(1, int(0.2*120))),
                    elitism_k=max(2, int(0.02*120)),
                    num_chargers=system_params['M'],
                    verbose=True)

    # Print results
    print("\n=== GA Result Summary ===")
    print("Objective J (normalized weighted):", result['J'])
    print("F1 (cost, raw) :", result['F1'], "F1_norm:", result['F1_norm'])
    print("F2 (deg, raw):", result['F2'], "F2_norm:", result['F2_norm'])
    print("F3 (var, raw):", result['F3'], "F3_norm:", result['F3_norm'])
    print("F4 (satisfaction, raw):", result['F4'], "F4_norm:", result['F4_norm'])
    print("Penalties (raw):", result['Omega_raw'], "Penalties_norm:", result['Omega_norm'])
    print("Violations (SoC, occ, grid):", result['V_SoC'], result['V_occ'], result['V_grid'])
    print("Generations executed:", result['generations_executed'])

    # Save best schedule
    best = result['best_schedule']
    df = pd.DataFrame(best, index=[ev['id'] for ev in admitted_evs])
    csv_path = os.path.abspath('best_schedule_normalized.csv')
    df.to_csv(csv_path, index=True)
    print(f"Saved best_schedule_normalized.csv to: {csv_path}")

    # Per-EV diagnostics
    print("\n--- Per-EV Diagnostics (admitted) ---")
    for i, ev in enumerate(admitted_evs):
        delivered = float(np.sum(best[i, ev['T_arr_idx']:ev['T_dep_idx']]) * delta_t)
        SoC_T = ev['SoC_init'] + (delivered / ev['Ecap'] if ev['Ecap'] > 0 else 0.0)
        Si = compute_Si_from_schedule(best, admitted_evs, delta_t)[i]
        print(f"EV{ev['id']}: delivered={delivered:.2f} kWh, SoC_T={SoC_T:.3f}, S_i={Si:.3f}")

if __name__ == "__main__":
    main()
