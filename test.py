#!/usr/bin/env python3
"""
pipeline_fixed.py

Same integrated pipeline as before but with KeyError fixes:
 - safe retrieval of P_ref / R_i (no direct ev['R_i'] indexing)
 - ensure admitted EVs carry 'P_ref'
 - compute_Si_from_schedule uses safe pref lookup

Other functionality unchanged.
"""

import argparse
import random
import math
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

# -------------------------
# Load EV pool
# -------------------------

def load_ev_pool(path: str) -> List[Dict]:
    path = os.path.abspath(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        evs = []
        for _, row in df.iterrows():
            ev = {
                'id': int(row['id']),
                'Ecap': float(row['Ecap']),
                'SoC_init': float(row['SoC_init']),
                'SoC_max': float(row['SoC_max']),
                'R_i': float(row['R_i']) if 'R_i' in row and not pd.isna(row['R_i']) else None,
                'T_stay': float(row['T_stay'])
            }
            if 'cdeg' in row and not pd.isna(row['cdeg']):
                ev['cdeg'] = float(row['cdeg'])
            if 'SoC_min' in row and not pd.isna(row['SoC_min']):
                ev['SoC_min'] = float(row['SoC_min'])
            evs.append(ev)
        return evs
    elif path.lower().endswith(".json"):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return data
    else:
        raise ValueError("Unsupported file type. Use .csv or .json")

def default_ev_pool() -> List[Dict]:
    return [
        {'id': 1, 'Ecap': 60.0, 'SoC_init': 0.30, 'SoC_max': 0.80, 'R_i': 7.0, 'T_stay': 4.0, 'cdeg': 0.02},
        {'id': 2, 'Ecap': 40.0, 'SoC_init': 0.60, 'SoC_max': 0.90, 'R_i': 11.0,'T_stay': 1.0, 'cdeg': 0.02},
        {'id': 3, 'Ecap': 50.0, 'SoC_init': 0.20, 'SoC_max': 0.60, 'R_i': 7.0, 'T_stay': 6.0, 'cdeg': 0.02},
    ]

# -------------------------
# Stage-I: priority scheduling helpers
# -------------------------

def calc_delta_E(ev: Dict) -> float:
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    """
    Eq. (9) with safe P_ref retrieval.
    If P_ref (or R_i) absent, apply raw ΔE/T_stay (no cap).
    """
    deltaE = calc_delta_E(ev)
    if ev['T_stay'] <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / ev['T_stay']
    pref = ev.get('R_i', ev.get('P_ref', None))
    if pref is None or pref <= 0:
        return raw
    return min(raw, pref)

def calc_phi(ev: Dict) -> float:
    """Eq. (10): phi = min(1, p_req / P_ref) with safe pref retrieval"""
    p_req = calc_p_req(ev)
    pref = ev.get('R_i', ev.get('P_ref', None))
    if pref is None or pref <= 0:
        return 0.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], cdeg: float) -> Dict[int, float]:
    values = [ (ev.get('cdeg', cdeg) * calc_delta_E(ev)) for ev in ev_list ]
    vmin = min(values)
    vmax = max(values)
    denom = (vmax - vmin) if (vmax != vmin) else 1.0
    return {ev['id']: ( (ev.get('cdeg', cdeg) * calc_delta_E(ev)) - vmin) / denom for ev in ev_list}

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = P_max - P_avg if P_max != P_avg else 1.0
    return max(0.0, (p_hat - P_avg) / denom)

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
                         pi_rev_min: float, pi_rev_max: float) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}
    D_map = calc_degradation_factor(ev_list, cdeg)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)
    G_factor = min(1.0, G_factor)
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
            'raw_deg': ev.get('cdeg', cdeg) * calc_delta_E(ev)
        }
    return lambda_scores, details

def assign_chargers(lambda_scores: Dict[int, float], M: int) -> List[int]:
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    return [eid for eid, _ in sorted_evs[:M]]

# -------------------------
# Stage-II helpers and objectives
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
        # SAFE pref retrieval: prefer P_ref, then R_i, else None -> treat as no-cap
        pref = ev.get('P_ref', ev.get('R_i', None))
        if pref is None or pref <= 0:
            phi_i = 0.0
        else:
            phi_i = min(1.0, preq / pref)
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

# -------------------------
# GA helpers (seeding, repair, fitness)
# -------------------------

def build_bounds_and_mask(evs, M, T):
    lower = [0.0] * (M * T)
    upper = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        pmin = -ev.get('P_ref', ev.get('R_i', 0.0))
        pmax = ev.get('P_ref', ev.get('R_i', 0.0))
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
    fixed_mask = [abs(l - u) < 1e-12 for l, u in zip(lower, upper)]
    return lower, upper, fixed_mask

def seed_individual_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25):
    chrom = [0.0] * (M * T)
    price_arr = np.array(pi_buy)
    for i in range(M):
        ev = evs[i]
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        Tstay = ev['T_stay'] if ev['T_stay'] > 0 else 1e-9
        raw_preq = DeltaE / Tstay
        preq = min(raw_preq, ev.get('P_ref', ev.get('R_i', raw_preq)))
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            continue
        stay_slots = list(range(Tarr, Tdep))
        if len(stay_slots) == 0:
            continue
        prices_in_stay = price_arr[stay_slots]
        k = max(1, int(math.ceil(seed_fraction * len(stay_slots))))
        chosen_idx_rel = list(np.argsort(prices_in_stay)[:k])
        for rel in chosen_idx_rel:
            t = stay_slots[rel]
            chrom[flatten_index(i, t, T)] = preq
    return chrom

def seed_individual_full_charge(evs, M, T, delta_t):
    chrom = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        slots = max(1, Tdep - Tarr)
        if slots <= 0:
            continue
        per_slot_power = (DeltaE / (slots * delta_t)) if (slots * delta_t) > 0 else 0.0
        per_slot_power = min(per_slot_power, ev.get('P_ref', ev.get('R_i', per_slot_power)))
        for t in range(Tarr, Tdep):
            chrom[flatten_index(i, t, T)] = per_slot_power
    return chrom

def repair_individual(individual, evs, M, T, delta_t):
    arr = unflatten(individual, M, T)
    for i in range(M):
        ev = evs[i]
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        Ecap = ev['Ecap']
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * Ecap)
        p_slice = arr[i, Tarr:Tdep].copy()
        E_del = float(np.sum(np.maximum(p_slice, 0.0)) * delta_t)
        if abs(E_del - DeltaE) < 1e-6:
            pass
        elif E_del < DeltaE and np.any(p_slice < ev.get('P_ref', ev.get('R_i', 0.0)) - 1e-9):
            need_kwh = DeltaE - E_del
            need_kw = need_kwh / delta_t
            headroom = np.maximum(ev.get('P_ref', ev.get('R_i', 0.0)) - p_slice, 0.0)
            total_head = float(np.sum(headroom))
            if total_head > 1e-9:
                scale = min(1.0, need_kw / total_head)
                p_slice += headroom * scale
        elif E_del > DeltaE and np.any(p_slice > 1e-9):
            factor = DeltaE / E_del if E_del > 0 else 0.0
            p_slice = p_slice * factor
        arr[i, Tarr:Tdep] = p_slice
        arr[i, :] = np.clip(arr[i, :], -ev.get('P_ref', ev.get('R_i', 0.0)), ev.get('P_ref', ev.get('R_i', 0.0)))
        for t in range(0, Tarr):
            arr[i,t] = 0.0
        for t in range(Tdep, T):
            arr[i,t] = 0.0
    individual[:] = arr.reshape(M*T).tolist()
    return individual

def make_fitness_function(evs, M, T, delta_t,
                          pi_buy, pi_rev,
                          cdeg_arr,
                          P_max, num_chargers,
                          w1, w2, w3, w4,
                          alpha1, alpha2, alpha3):
    total_DeltaE = sum(max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap']) for ev in evs)
    max_price = max(max(pi_buy) if hasattr(pi_buy, '__iter__') else pi_buy,
                    max(pi_rev) if hasattr(pi_rev, '__iter__') else pi_rev)
    sum_Pref = sum(ev.get('P_ref', ev.get('R_i', 0.0)) for ev in evs)
    denom_F1 = max_price * total_DeltaE + 1e-9
    denom_F2 = max(cdeg_arr) * total_DeltaE + 1e-9
    denom_F3 = (sum_Pref ** 2) * T + 1e-9
    denom_F4 = (M + 1e-9)
    denom_Omega = (total_DeltaE + 1e-9)

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
# GA runner
# -------------------------

def run_ga(evs, T, delta_t,
           pi_buy, pi_rev,
           P_max,
           weights,
           alpha1, alpha2, alpha3,
           pop_size=120, ngen=300,
           cxpb=0.9, mutpb=0.3, eta_c=20.0, eta_m=20.0,
           tournament_size=3,
           stagnation_generations=80,
           seed_count=20,
           elitism_k=4,
           verbose=True):
    random.seed(42)
    M = len(evs)
    for ev in evs:
        if 'P_ref' not in ev:
            ev['P_ref'] = ev.get('R_i', 0.0)
    cdeg_arr = [ev.get('cdeg', 0.02) for ev in evs]
    num_chargers = len(evs)
    w1 = weights.get('w1', 0.15)
    w2 = weights.get('w2', 0.15)
    w3 = weights.get('w3', 0.10)
    w4 = weights.get('w4', 0.60)

    lower, upper, fixed_mask = build_bounds_and_mask(evs, M, T)
    safe_low = list(lower)
    safe_up = list(upper)
    for idx in range(M*T):
        if abs(lower[idx] - upper[idx]) < 1e-12:
            safe_up[idx] = safe_low[idx] + 1e-9

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def attr_float_at_index(idx):
        lo = lower[idx]
        up = upper[idx]
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

    def safe_mate(ind1, ind2):
        tools.cxSimulatedBinaryBounded(ind1, ind2, low=safe_low, up=safe_up, eta=eta_c)
        for idx in range(M*T):
            if fixed_mask[idx]:
                ind1[idx] = lower[idx]
                ind2[idx] = lower[idx]
        return (ind1, ind2)

    def safe_mutate(individual):
        tools.mutPolynomialBounded(individual, low=safe_low, up=safe_up, eta=eta_m, indpb=1.0/(M*T))
        for idx in range(M*T):
            if fixed_mask[idx]:
                individual[idx] = lower[idx]
        return (individual,)

    toolbox.register("mate", safe_mate)
    toolbox.register("mutate", safe_mutate)

    pop = toolbox.population(n=pop_size)
    num_seed = min(seed_count, pop_size)
    if num_seed >= 1:
        pop[0][:] = seed_individual_full_charge(evs, M, T, delta_t)
    if num_seed >= 2:
        pop[1][:] = seed_individual_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25)
    for s in range(2, num_seed):
        seed = seed_individual_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25)
        for idx in range(M*T):
            if not fixed_mask[idx]:
                seed[idx] += random.uniform(-0.05, 0.05) * (upper[idx] - lower[idx])
                seed[idx] = max(lower[idx], min(upper[idx], seed[idx]))
        pop[s][:] = seed

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

        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            if random.random() <= cxpb:
                toolbox.mate(offspring[i], offspring[i + 1])
                repair_individual(offspring[i], evs, M, T, delta_t)
                repair_individual(offspring[i+1], evs, M, T, delta_t)

        for i in range(len(offspring)):
            if random.random() <= mutpb:
                toolbox.mutate(offspring[i])
                repair_individual(offspring[i], evs, M, T, delta_t)

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

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
                        help='Path to EV definitions file (CSV or JSON). If omitted, built-in 3-EV example is used.')
    parser.add_argument('--chargers', type=int, default=None, help='Number of chargers (M)')
    parser.add_argument('--t_slots', type=int, default=48, help='Number of time slots (T)')
    parser.add_argument('--delta_t', type=float, default=0.25, help='slot length in hours')
    parser.add_argument('--population', type=int, default=120)
    parser.add_argument('--generations', type=int, default=300)
    args = parser.parse_args()

    if args.ev_file:
        ev_pool = load_ev_pool(args.ev_file)
        print(f"Loaded {len(ev_pool)} EVs from {args.ev_file}")
    else:
        found = False
        for fname in ['evs.csv', 'evs_30.csv', 'evs_100.csv']:
            if os.path.exists(fname):
                ev_pool = load_ev_pool(fname)
                print(f"No --ev-file provided; found '{fname}' in current directory and will use it.")
                print(f"Loaded {len(ev_pool)} EVs from {fname}")
                found = True
                break
        if not found:
            ev_pool = default_ev_pool()
            print("No EV file found; using built-in example (3 EVs).")

    N = len(ev_pool)
    if args.chargers is not None:
        M = args.chargers
    else:
        M = max(1, min(N, max(1, N // 3)))
    print("Number of chargers M set to:", M)

    system_params = {
        'M': M,
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

    lambda_scores, details = calc_priority_scores(
        ev_pool,
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

    admitted_ids = assign_chargers(lambda_scores, M)
    admitted = [ev for ev in ev_pool if ev['id'] in admitted_ids]

    # Stage-I print
    print("\n=== Stage-I Priority Scheduling ===\n")
    print("Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):")
    for ev in ev_pool:
        dE = details[ev['id']]['DeltaE']
        p_req = details[ev['id']]['p_req']
        phi = details[ev['id']]['phi']
        print(f"EV{ev['id']}: DeltaE = {dE:.3f} kWh, p_req = {p_req:.3f} kW, phi = {phi:.3f}")
    print()
    print("Step 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):")
    for ev in ev_pool:
        raw_deg = details[ev['id']]['raw_deg']
        D = details[ev['id']]['Dfactor']
        print(f"EV{ev['id']}: cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {D:.3f}")
    print()
    G_factor = next(iter(details.values()))['Gfactor']
    print("Step 3 — Grid stress factor (Gfactor) (Eq. (24)):")
    print(f"Gfactor (capped <=1.0) = {G_factor:.4f}")
    print()
    P_factor = next(iter(details.values()))['Pfactor']
    print("Step 4 — Price factor (Pfactor) (Eq. (25),(26)):")
    print(f"Pfactor = {P_factor:.3f}\n")
    print("Step 5 — Priority scores (lambda_i) (Eq. (22)):")
    for ev in ev_pool:
        lam = details[ev['id']]['lambda']
        print(f"lambda_{ev['id']} = {lam:.5f}")
    print()
    sorted_by_lambda = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    print("Step 6 — Ranking and Admission:")
    rank_str = " > ".join([f"EV{eid}" for eid, _ in sorted_by_lambda])
    print(f"Ranking (desc lambda): {rank_str}")
    print(f"With M={system_params['M']} chargers, admitted EVs: {', '.join('EV'+str(x) for x in admitted_ids)}")
    waiting = [ev['id'] for ev in ev_pool if ev['id'] not in admitted_ids]
    print(f"Waiting EVs: {', '.join('EV'+str(x) for x in waiting)}")
    print("\nStage-I admitted EV ids:", admitted_ids)

    # Convert admitted EVs for Stage-II
    T = args.t_slots
    delta_t = args.delta_t
    admitted_evs = []
    for ev in admitted:
        slots = max(1, int(math.ceil(ev['T_stay'] / delta_t)))
        dep_idx = min(T, slots)
        # SAFE P_ref assignment: prefer R_i, else keep 0 if missing (repair will use 0 cap)
        p_ref = ev.get('R_i', ev.get('P_ref', 0.0))
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'P_ref': p_ref,
            'T_stay': ev['T_stay'],
            'T_arr_idx': 0,
            'T_dep_idx': dep_idx,
            'cdeg': ev.get('cdeg', system_params['cdeg'])
        })

    # GA inputs
    pi_buy_arr = [system_params['pi_buy']] * T
    pi_rev_arr = [system_params['pi_rev']] * T
    P_max = system_params['P_max']
    weights_stage2 = {'w1': 0.15, 'w2': 0.15, 'w3': 0.10, 'w4': 0.60}
    alpha1, alpha2, alpha3 = 200.0, 100.0, 100.0

    result = run_ga(evs=admitted_evs,
                    T=T, delta_t=delta_t,
                    pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
                    P_max=P_max,
                    weights=weights_stage2,
                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                    pop_size=args.population, ngen=args.generations,
                    cxpb=0.9, mutpb=0.35,
                    tournament_size=3,
                    stagnation_generations=80,
                    seed_count=max(10, int(0.15 * args.population)),
                    elitism_k=max(2, int(0.03 * args.population)),
                    verbose=True)

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

    print("\n--- Per-EV Diagnostics (admitted) ---")
    for i, ev in enumerate(admitted_evs):
        delivered_kwh = float(np.sum(np.maximum(best[i,:], 0.0)) * delta_t)
        SoC = ev['SoC_init']
        for t in range(T):
            SoC += (best[i,t] * delta_t) / ev['Ecap']
        SoC_T = SoC
        Si_list = result.get('Si_list', [])
        Si = Si_list[i] if i < len(Si_list) else None
        print(f"EV{ev['id']}: delivered={delivered_kwh:.2f} kWh, SoC_T={SoC_T:.3f}, S_i={Si:.3f}")

    # Derived metrics
    delivered = np.sum(np.maximum(best, 0.0), axis=1) * delta_t
    avg_cost_per_ev = result['F1'] / (len(admitted_evs) if len(admitted_evs) > 0 else 1)
    print("\n--- Derived Metrics ---")
    print("Average net energy cost per admitted EV:", avg_cost_per_ev)
    if delivered.size > 0:
        arr = np.array(np.sort(delivered))
        n = arr.size
        if n > 0 and np.sum(arr) > 1e-9:
            index = np.arange(1, n+1)
            gini = (np.sum((2*index - n - 1) * arr)) / (n * np.sum(arr) + 1e-9)
            print("Fairness (Gini on delivered kWh):", abs(gini))
    print("Done.")

if __name__ == "__main__":
    main()
