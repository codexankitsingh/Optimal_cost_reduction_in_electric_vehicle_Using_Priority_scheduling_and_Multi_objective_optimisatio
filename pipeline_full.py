"""
both the part of priority scheduling and GA part integrated here
"""

import random
import math
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# DEAP imports
from deap import base, creator, tools

# -------------------------
# Stage-I: Priority Scheduling (your code wrapped)
# -------------------------

def calc_delta_E(ev: Dict) -> float:
    """Eq. (7): Delta E_i = (SoC_max - SoC_init) * Ecap (kWh)"""
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    """Eq. (9) with cap at R_i to match paper example"""
    deltaE = calc_delta_E(ev)
    if ev['T_stay'] <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / ev['T_stay']
    return min(raw, ev['R_i'])

def calc_phi(ev: Dict) -> float:
    """Eq. (10): phi = min(1, p_req / P_ref)"""
    p_req = calc_p_req(ev)
    pref = ev['R_i']
    if pref <= 0:
        return 0.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], cdeg: float) -> Dict[int, float]:
    """Eq. (23): min-max normalization of cdeg * DeltaE"""
    values = [cdeg * calc_delta_E(ev) for ev in ev_list]
    vmin, vmax = min(values), max(values)
    denom = vmax - vmin if vmax != vmin else 1.0
    return {ev['id']: (cdeg * calc_delta_E(ev) - vmin) / denom for ev in ev_list}

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    """Eq. (24): G_factor_t = max(0, (P_hat_agg - P_avg) / (P_max - P_avg))"""
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = P_max - P_avg if P_max != P_avg else 1.0
    return max(0.0, (p_hat - P_avg) / denom)

def calc_price_factor(pi_buy: float, pi_rev: float,
                      pi_buy_min: float, pi_buy_max: float,
                      pi_rev_min: float, pi_rev_max: float) -> float:
    """Eq. (25)-(26): normalized buy/rev then P_factor = P_buy - P_rev"""
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
    """Compute phi, Dfactor, Gfactor, Pfactor and lambda for each EV (Eq. (22))."""
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}
    D_map = calc_degradation_factor(ev_list, cdeg)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)
    P_factor = calc_price_factor(pi_buy, pi_rev,
                                 pi_buy_min, pi_buy_max,
                                 pi_rev_min, pi_rev_max)

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
    """Rank EVs by descending lambda. Tie-breaker: lower EV id first."""
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    return [eid for eid, _ in sorted_evs[:M]]

def run_stage1(ev_list: List[Dict], system_params: Dict) -> Tuple[List[Dict], Dict]:
    """Runs Stage-I and prints intermediate results (keeps same prints as original)."""
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

    # Print same-format output for traceability
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
        raw_deg = system_params['cdeg'] * details[ev['id']]['DeltaE']
        D = details[ev['id']]['Dfactor']
        print(f"EV{ev['id']}: cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {D:.3f}")
    print()
    G_factor = next(iter(details.values()))['Gfactor']
    P_factor = next(iter(details.values()))['Pfactor']
    print("Step 3 — Grid stress factor (Gfactor) (Eq. (24)):")
    print(f"Gfactor = {G_factor:.4f}")
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
# Stage-II: GA-based power scheduling (DEAP) with normalization & elitism
# -------------------------

def flatten_index(i, t, T): return i * T + t
def unflatten(individual, M, T): return np.array(individual, dtype=float).reshape((M, T))

def compute_F1(net_schedule, pi_buy, pi_rev, delta_t):
    """Eq.(17): Net energy cost (raw dollars)"""
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
    """Eq.(18): Degradation cost (raw dollars)"""
    M, T = net_schedule.shape
    return sum(cdeg_arr[i] * abs(net_schedule[i, t]) * delta_t for i in range(M) for t in range(T))

def compute_F3(net_schedule):
    """Eq.(19): Grid load variance (raw squared units)"""
    L = np.sum(net_schedule, axis=0)
    Lbar = np.mean(L)
    return np.sum((L - Lbar) ** 2)

def compute_Si_from_schedule(net_schedule, evs, delta_t):
    """Compute per-EV satisfaction Si depending on final SoC (Eq.12-based)"""
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
    """F4: total satisfaction (raw sum of Si)"""
    Si = compute_Si_from_schedule(net_schedule, evs, delta_t)
    return sum(Si), Si

def compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t):
    """
    Raw penalty components:
    - V_SoC: cumulative magnitude of SoC violations across time/Evs
    - V_occ: sum over t of excess active chargers beyond capacity
    - V_grid: sum over t of (L_t - P_max) positive part
    """
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
    """
    Returns evaluator that computes normalized J and caches breakdown.
    Normalization constants precomputed here for consistent scaling.
    """
    # Precompute normalization denominators (theoretical upper bounds)
    max_power_sum = sum(ev['P_ref'] for ev in evs) if evs else 1.0  # kW
    max_energy_over_horizon = max_power_sum * delta_t * T  # kWh
    max_price = max(max(pi_buy) if hasattr(pi_buy, '__len__') else pi_buy,
                    max(pi_rev) if hasattr(pi_rev, '__len__') else pi_rev)
    max_cdeg = max(cdeg_arr) if len(cdeg_arr) > 0 else 1.0
    # For variance normalization: worst-case approx (max_power_sum^2 * T)
    denom_F1 = (max_price * max_energy_over_horizon + 1e-9)
    denom_F2 = (max_cdeg * max_energy_over_horizon + 1e-9)
    denom_F3 = ((max_power_sum ** 2) * T + 1e-9)
    denom_F4 = (M + 1e-9)
    denom_Omega = (max_power_sum * delta_t * T + 1e-9)  # normalizer for penalty sum

    def evaluate(individual):
        net_schedule = unflatten(individual, M, T)
        F1 = compute_F1(net_schedule, pi_buy, pi_rev, delta_t)
        F2 = compute_F2(net_schedule, cdeg_arr, delta_t)
        F3 = compute_F3(net_schedule)
        F4_sum, Si_list = compute_F4(net_schedule, evs, delta_t)
        V_SoC, V_occ, V_grid = compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t)

        # Raw penalty
        Omega_raw = alpha1 * V_SoC + alpha2 * V_occ + alpha3 * V_grid

        # Normalize each objective (theoretical bounds)
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
    """Return lower/upper arrays; outside availability windows set to fixed 0."""
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
    """Place p_req on lowest-price fraction of slots inside each EV stay to seed population."""
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
    """
    GA orchestration (normalized multi-objective scalarization).
    Returns result dict with best schedule and objective breakdown.
    """

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

    # DEAP creator guards (avoid repeated creation errors)
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

    # evaluation & genetic ops
    eval_fn = make_fitness_function(evs, M, T, delta_t,
                                    pi_buy, pi_rev,
                                    cdeg_arr,
                                    P_max, num_chargers,
                                    w1, w2, w3, w4,
                                    alpha1, alpha2, alpha3)
    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Prepare bounded operators with safety for fixed genes
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

    # initialize population with seeding
    pop = toolbox.population(n=pop_size)
    num_seed = min(seed_count, pop_size)
    for s in range(num_seed):
        seed_chrom = seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25)
        pop[s][:] = seed_chrom

    # evaluate initial pop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # track best & elitism
    best = tools.selBest(pop, 1)[0]
    best_J = best.fitness.values[0]
    no_improve = 0
    gen = 0
    if verbose:
        print(f"GA start: pop_size={pop_size}, ngen={ngen}, M={M}, T={T}")
        print("Initial best J =", round(best_J, 8))

    while gen < ngen and no_improve < stagnation_generations:
        gen += 1
        # selection + clone
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

        # elitism: keep top-k from current pop into next generation
        elites = tools.selBest(pop, elitism_k)
        # build new pop: combine offspring and elites, then select best pop_size individuals
        combined = offspring + list(map(toolbox.clone, elites))
        # ensure validity of fitness
        for ind in combined:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        # select best individuals for next generation
        combined.sort(key=lambda ind: ind.fitness.values[0])
        pop = combined[:pop_size]

        # update best
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

    # final unpack
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
    # EV pool (Stage-I example)
    ev_pool = [
        {'id': 1, 'Ecap': 60.0, 'SoC_init': 0.30, 'SoC_max': 0.80, 'R_i': 7.0, 'T_stay': 4.0},
        {'id': 2, 'Ecap': 40.0, 'SoC_init': 0.60, 'SoC_max': 0.90, 'R_i': 11.0, 'T_stay': 1.0},
        {'id': 3, 'Ecap': 50.0, 'SoC_init': 0.20, 'SoC_max': 0.60, 'R_i': 7.0, 'T_stay': 6.0},
    ]

    system_params = {
        'M': 2,
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

    # Stage-I
    admitted, details = run_stage1(ev_pool, system_params)
    print("\nStage-I admitted EV ids:", [ev['id'] for ev in admitted])

    # Convert admitted EVs for Stage-II
    T = 48
    delta_t = 0.25
    admitted_evs = []
    for ev in admitted:
        slots = max(1, int(math.ceil(ev['T_stay'] / delta_t)))
        dep_idx = min(T, slots)
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'P_ref': ev['R_i'],
            'T_stay': ev['T_stay'],
            'T_arr_idx': 0,
            'T_dep_idx': dep_idx,
            'cdeg': 0.02
        })

    # GA inputs and hyperparams (tune these as needed)
    pi_buy_arr = [0.25] * T
    pi_rev_arr = [0.18] * T
    P_max = 25.0
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    alpha1, alpha2, alpha3 = 50.0, 50.0, 50.0  # scaled penalties (smaller after normalization)

    # Run Stage-II GA (modified defaults: larger ngen/pop are fine to increase if you have time)
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

    # Print and save outputs
    print("\n=== GA Result Summary ===")
    print("Objective J (normalized weighted):", result['J'])
    print("F1 (cost, raw) :", result['F1'], "F1_norm:", result['F1_norm'])
    print("F2 (degradation, raw):", result['F2'], "F2_norm:", result['F2_norm'])
    print("F3 (variance, raw):", result['F3'], "F3_norm:", result['F3_norm'])
    print("F4 (satisfaction, raw):", result['F4'], "F4_norm:", result['F4_norm'])
    print("Penalties (raw):", result['Omega_raw'], "Penalties_norm:", result['Omega_norm'])
    print("Violations (SoC, occ, grid):", result['V_SoC'], result['V_occ'], result['V_grid'])
    print("Generations executed:", result['generations_executed'])

    # Save best schedule as CSV (rows: EV id, cols: slot indices)
    best = result['best_schedule']
    df = pd.DataFrame(best, index=[ev['id'] for ev in admitted_evs])
    csv_path = os.path.abspath('best_schedule_normalized.csv')
    df.to_csv(csv_path, index=True)
    print(f"Saved best_schedule_normalized.csv to: {csv_path}")

if __name__ == "__main__":
    main()
