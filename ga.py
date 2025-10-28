"""
Ga implementation
Implements fitness J = w1*F1 + w2*F2 + w3*F3 - w4*F4 + Omega(X)
F1..F4 and Omega follow the paper's equations (see uploaded paper). 
"""

import random
import math
import copy
import numpy as np
import pandas as pd

# DEAP imports
from deap import base, creator, tools, algorithms

# -----------------------
# Utility / mapping helpers
# -----------------------

def flatten_index(i, t, T):
    """Map (i,t) to flat index in chromosome of length M*T."""
    return i * T + t

def unflatten(individual, M, T):
    """Return a 2D numpy array shape (M, T) from flattened individual list."""
    arr = np.array(individual, dtype=float)
    return arr.reshape((M, T))

# -----------------------
# Objective / constraint helpers (F1..F4, Omega)
# -----------------------

def compute_F1(net_schedule, pi_buy, pi_rev, delta_t):
    """
    Eq.(17): Net energy cost.
    net_schedule: M x T array with power p_{i,t} (kW) (positive charge, negative discharge)
    pi_buy, pi_rev: arrays length T ($/kWh)
    delta_t: slot duration (hours)
    Returns: scalar cost (dollars)
    """
    M, T = net_schedule.shape
    cost = 0.0
    for t in range(T):
        for i in range(M):
            p = net_schedule[i, t]
            if p > 0:
                cost += pi_buy[t] * p * delta_t
            elif p < 0:
                # pi_rev[t] * min(p,0) — since p<0, produces negative sign = revenue
                cost += pi_rev[t] * p * delta_t
            # if p == 0: nothing
    return cost

def compute_F2(net_schedule, cdeg_arr, delta_t):
    """
    Eq.(18): Battery degradation cost.
    cdeg_arr: array of length M (per-EV degradation $/kWh)
    """
    M, T = net_schedule.shape
    deg = 0.0
    for i in range(M):
        for t in range(T):
            deg += cdeg_arr[i] * abs(net_schedule[i, t]) * delta_t
    return deg

def compute_F3(net_schedule):
    """
    Eq.(19): Grid load variance across T slots: sum_t (L_t - L_bar)^2,
    where L_t = sum_i p_{i,t}
    """
    M, T = net_schedule.shape
    L = np.sum(net_schedule, axis=0)  # length-T
    Lbar = np.mean(L)
    var = np.sum((L - Lbar) ** 2)
    return var

def compute_Si_from_schedule(net_schedule, evs, delta_t):
    """
    Compute per-EV satisfaction Si using Eq.(12) which depends on final SoC SoC_{i,T}.
    Steps:
      - simulate SoC evolution from arrival slot to departure slot using p_{i,t}
      - compute δi and then Si = max(0, 1 - phi_i * δ_i) where phi_i is sensitivity computed from Stage-I P_req/Pref
    Returns: list of Si for each EV (length M)
    """
    M, T = net_schedule.shape
    Si_list = [0.0] * M

    for idx in range(M):
        ev = evs[idx]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']  # initial SoC fraction
        SoC_min = ev.get('SoC_min', 0.0)
        SoC_max = ev['SoC_max']

        # simulate through all T slots; genes outside [arr,dep) are expected to be zero
        for t in range(T):
            p = net_schedule[idx, t]
            # increment of energy fraction = (p * delta_t) / Ecap
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
            # keep SoC unconstrained here; penalty will handle violations
        # final SoC
        SoC_T = SoC

        # Step 1-3: compute Ereq and phi (sensitivity)
        Ereq = max(0.0, (SoC_max - ev['SoC_init']) * Ecap)
        Tstay = ev['T_stay']
        if Tstay > 0:
            preq = Ereq / Tstay
        else:
            preq = float('inf') if Ereq > 0 else 0.0
        phi_i = min(1.0, preq / ev['P_ref']) if ev['P_ref'] > 0 else 0.0

        # Step 4: delta_i = 0 if satisfied, else normalized shortfall
        if SoC_T >= SoC_max:
            delta_i = 0.0
        else:
            denom = (SoC_max - ev['SoC_init'])
            if denom <= 0.0:
                delta_i = 1.0  # degenerate: treat as worst
            else:
                delta_i = (SoC_max - SoC_T) / denom
                # ensure non-negative
                if delta_i < 0.0:
                    delta_i = 0.0

        # Step 5: satisfaction
        Si = max(0.0, 1.0 - phi_i * delta_i)
        Si_list[idx] = Si

    return Si_list

def compute_F4(net_schedule, evs, delta_t):
    """
    F4 is total satisfaction sum_i S_i (Eq. (13)) — higher is better.
    We will return F4 as sum Si.
    """
    Si = compute_Si_from_schedule(net_schedule, evs, delta_t)
    return sum(Si), Si  # return per-ev list as well

def compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t):
    """
    Omega(X) per Eq.(29): alpha1 * sum SoC violations + alpha2 * sum occ violations + alpha3 * sum grid violations
    This function returns the three raw violation sums (V_SoC, V_occ, V_grid).
    - V_SoC: sum of absolute amount by which SoC_t goes outside [SoC_min, SoC_max] across all times and EVs
      (we sum magnitude of violation across all time slots)
    - V_occ: sum_t max(0, number_active - chargers)
      here "active" means p != 0 (or > epsilon)
    - V_grid: sum_t max(0, sum_i p_{i,t} - P_max)
    """
    M, T = net_schedule.shape
    # compute SoC trajectory per EV and accumulate SoC violation across time
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
            # Check violation at this time t
            if SoC < SoC_min - eps:
                V_SoC += (SoC_min - SoC)
            elif SoC > SoC_max + eps:
                V_SoC += (SoC - SoC_max)
    # grid and occupancy violations per slot
    V_grid = 0.0
    V_occ = 0.0
    for t in range(T):
        Lt = float(np.sum(net_schedule[:, t]))
        if Lt > P_max:
            V_grid += (Lt - P_max)
        # occupancy: number of EVs with |p|>eps at time t
        active = int(np.sum(np.abs(net_schedule[:, t]) > 1e-6))
        if active > num_chargers:
            V_occ += (active - num_chargers)
    return V_SoC, V_occ, V_grid

# -----------------------
# Fitness wrapper
# -----------------------

def make_fitness_function(evs, M, T, delta_t,
                          pi_buy, pi_rev,
                          cdeg_arr,
                          P_max, num_chargers,
                          w1, w2, w3, w4,
                          alpha1, alpha2, alpha3):
    """
    Returns an evaluate(individual) function for DEAP that computes J and returns (J,)
    It also stores component values for later retrieval using attribute cache.
    """

    def evaluate(individual):
        # flatten -> net_schedule M x T
        net_schedule = unflatten(individual, M, T)
        # F1
        F1 = compute_F1(net_schedule, pi_buy, pi_rev, delta_t)
        # F2
        F2 = compute_F2(net_schedule, cdeg_arr, delta_t)
        # F3
        F3 = compute_F3(net_schedule)
        # F4
        F4_sum, Si_list = compute_F4(net_schedule, evs, delta_t)
        # Penalty raw violations
        V_SoC, V_occ, V_grid = compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t)
        Omega = alpha1 * V_SoC + alpha2 * V_occ + alpha3 * V_grid

        # Weighted objective J (Eq. (30))
        J = w1 * F1 + w2 * F2 + w3 * F3 - w4 * F4_sum + Omega

        # store breakdown on individual for later reporting (DEAP permits attributes)
        # We'll set them as attrs on the individual object (list is mutable)
        individual._cached = {
            'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4_sum,
            'Si_list': Si_list,
            'V_SoC': V_SoC, 'V_occ': V_occ, 'V_grid': V_grid,
            'Omega': Omega, 'J': J
        }
        return (J,)

    return evaluate


# Initialization: bounds & seeding

def build_bounds_and_mask(evs, M, T):
    """
    Build lower and upper arrays (length M*T) to pass to SBX and mutation operators.
    For times outside [T_arr_idx, T_dep_idx), set lower==upper==0 to force gene to stay 0.
    Also produce index->(i,t) mapping arrays if needed.
    Assumes ev fields:
      - 'T_arr_idx' and 'T_dep_idx' are integer indices in range [0, T) or [0,T] with dep exclusive.
    """
    lower = [0.0] * (M * T)
    upper = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        pmin = -ev['P_ref']
        pmax = ev['P_ref']
        # arrival and departure indices (if not provided, assume full horizon available)
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)  # exclusive
        for t in range(T):
            idx = flatten_index(i, t, T)
            if t >= Tarr and t < Tdep:
                lower[idx] = pmin
                upper[idx] = pmax
            else:
                # outside stay => force zero
                lower[idx] = 0.0
                upper[idx] = 0.0
    return lower, upper

def seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25):
    """
    Create a seeded chromosome as per Stage-I logic:
     - compute p_req (min(DeltaE/Tstay, P_ref)) for each EV
     - assign p_req in the lowest-price fraction of time slots within EV's stay
     - other in-stay slots get 0
    seed_fraction: fraction of the EV's stay slots to place charging (e.g., 0.25)
    """
    chrom = [0.0] * (M * T)
    T_total = T
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
        # select lowest-price slots within stay to place charging (seed)
        prices_in_stay = price_arr[stay_slots]
        k = max(1, int(math.ceil(seed_fraction * len(stay_slots))))
        # choose indices of k smallest prices
        chosen_idx_rel = list(np.argsort(prices_in_stay)[:k])
        for rel in chosen_idx_rel:
            t = stay_slots[rel]
            chrom[flatten_index(i, t, T)] = preq
    return chrom

# -----------------------
# GA runner


def run_ga(evs, admitted_ids, # admitted ids optional (evs list is the admitted set normally)
           T, delta_t,
           pi_buy, pi_rev,
           P_max,
           weights, # dict w1..w4
           alpha1, alpha2, alpha3,
           pop_size=100, ngen=200,
           cxpb=0.9, mutpb=0.2, eta_c=20.0, eta_m=20.0,
           tournament_size=3,
           stagnation_generations=25,
           seed_count=5,
           verbose=True):
    """
    Main GA orchestration. Returns best schedule (M x T), best J, breakdown dict.
    - evs: list of M admitted EV dicts (must contain required fields like Ecap, SoC_init, SoC_max, P_ref, T_arr_idx, T_dep_idx, T_stay)
    - weights: dict with keys 'w1','w2','w3','w4' summing to 1 (or not, they are used directly)
    """

    random.seed(42)

    M = len(evs)
    # prepare arrays
    cdeg_arr = [ev.get('cdeg', 0.02) for ev in evs]  # fallback
    num_chargers = len(evs)  # Stage-II runs only on admitted EVs; chargers assigned earlier -> equal to M (or possibly less)
    w1 = weights.get('w1', 0.25)
    w2 = weights.get('w2', 0.25)
    w3 = weights.get('w3', 0.25)
    w4 = weights.get('w4', 0.25)

    # Build bounds with zeros outside availability
    lower, upper = build_bounds_and_mask(evs, M, T)
    # Convert to lists for DEAP bounded operators
    lower_arr = lower
    upper_arr = upper

    # DEAP setup: minimize J
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # attribute generator: uniform between lower[i] and upper[i]
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

    # evaluation function
    eval_fn = make_fitness_function(evs, M, T, delta_t,
                                    pi_buy, pi_rev,
                                    cdeg_arr,
                                    P_max, num_chargers,
                                    w1, w2, w3, w4,
                                    alpha1, alpha2, alpha3)
    toolbox.register("evaluate", eval_fn)

    # selection (tournament)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # --- Safe bounded operators to avoid division-by-zero when low==up ---
    _EPS = 1e-9  # tiny range for fixed genes (<< active threshold 1e-6)

    # create safe copies of bounds and a mask of fixed positions
    safe_low = list(lower_arr)
    safe_up  = list(upper_arr)
    fixed_mask = [False] * (M * T)
    for idx in range(M * T):
        if lower_arr[idx] == upper_arr[idx]:
            fixed_mask[idx] = True
            safe_up[idx] = safe_low[idx] + _EPS

    # wrapped SBX that restores fixed genes exactly after crossover
    def safe_mate(ind1, ind2):
        tools.cxSimulatedBinaryBounded(ind1, ind2, low=safe_low, up=safe_up, eta=eta_c)
        for idx in range(M * T):
            if fixed_mask[idx]:
                ind1[idx] = lower_arr[idx]
                ind2[idx] = lower_arr[idx]
        return (ind1, ind2)

    # wrapped polynomial mutation that restores fixed genes exactly after mutation
    def safe_mutate(individual):
        tools.mutPolynomialBounded(individual, low=safe_low, up=safe_up, eta=eta_m, indpb=1.0/(M*T))
        for idx in range(M * T):
            if fixed_mask[idx]:
                individual[idx] = lower_arr[idx]
        return (individual,)

    toolbox.register("mate", safe_mate)
    toolbox.register("mutate", safe_mutate)

    # --- population initialization with seeding ---
    pop = toolbox.population(n=pop_size)
    # overwrite some individuals with seeded chromosomes
    num_seed = min(seed_count, pop_size)
    for s in range(num_seed):
        seed_chrom = seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25)
        # ensure seed respects bounds (should by construction)
        pop[s][:] = seed_chrom

    # evaluate initial pop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # GA main loop with early-stopping on stagnation
    best = tools.selBest(pop, 1)[0]
    best_J = best.fitness.values[0]
    no_improve = 0

    gen = 0
    if verbose:
        print("GA start: pop_size={}, ngen={}, M={}, T={}".format(pop_size, ngen, M, T))
        print("Initial best J =", round(best_J, 6))

    while gen < ngen and no_improve < stagnation_generations:
        gen += 1
        # selection
        offspring = toolbox.select(pop, len(pop))
        # clone
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover on pairs
        for i in range(0, len(offspring), 2):
            if i+1 >= len(offspring):
                break
            if random.random() <= cxpb:
                toolbox.mate(offspring[i], offspring[i+1])
                # ensure no invalid genes outside bounds (operators are bounded but small numerical corrections)
                # mutate later

        # mutation
        for i in range(len(offspring)):
            if random.random() <= mutpb:
                toolbox.mutate(offspring[i])

        # evaluate offspring
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # replacement: next generation = offspring (generational)
        pop[:] = offspring

        # track best
        current_best = tools.selBest(pop, 1)[0]
        current_best_J = current_best.fitness.values[0]

        if current_best_J + 1e-12 < best_J:
            best = toolbox.clone(current_best)
            best_J = current_best_J
            no_improve = 0
            if verbose:
                print("Gen {} improved best J -> {:.6f}".format(gen, best_J))
        else:
            no_improve += 1
            if verbose and gen % 10 == 0:
                print("Gen {} best J {:.6f} (no improve {})".format(gen, best_J, no_improve))

    # final unpack
    best_schedule = unflatten(best, M, T)
    # cached breakdown
    breakdown = getattr(best, "_cached", None)
    if breakdown is None:
        # re-evaluate to fill cache
        best.fitness.values = toolbox.evaluate(best)
        breakdown = best._cached

    # packaging results
    result = {
        'best_schedule': best_schedule, # numpy array MxT
        'J': breakdown['J'],
        'F1': breakdown['F1'],
        'F2': breakdown['F2'],
        'F3': breakdown['F3'],
        'F4': breakdown['F4'],
        'Si_list': breakdown['Si_list'],
        'V_SoC': breakdown['V_SoC'],
        'V_occ': breakdown['V_occ'],
        'V_grid': breakdown['V_grid'],
        'Omega': breakdown['Omega'],
        'generations_executed': gen
    }

    return result

# -----------------------
# Example usage (if run directly)
# -----------------------
if __name__ == "__main__":
    # Minimal example: use the Stage-I admitted EVs from the paper (EV2, EV3)
    # We'll create ev dicts consistent with the paper's Table I and Stage-I admission result.
    T = 48
    delta_t = 0.25  # 15-min slots
    # price arrays (paper used constant values in the example; for real use provide arrays of length T)
    pi_buy = [0.25] * T
    pi_rev = [0.18] * T

    # admitted EVs (EV2, EV3 from Stage-I example). We provide required fields / indices.
    # The paper used Tstay in hours only; here we must set arrival/departure indices.
    # For demo, assume both arrived at slot 0 and depart at slot T (full horizon). In practice fill T_arr_idx & T_dep_idx.
    ev2 = {
        'id': 2,
        'Ecap': 40.0,
        'SoC_init': 0.60,
        'SoC_max': 0.90,
        'P_ref': 11.0,
        'T_stay': 1.0,
        'T_arr_idx': 0,
        'T_dep_idx': 4,    # e.g., 1 hour (4 slots)
        'cdeg': 0.02
    }
    ev3 = {
        'id': 3,
        'Ecap': 50.0,
        'SoC_init': 0.20,
        'SoC_max': 0.60,
        'P_ref': 7.0,
        'T_stay': 6.0,
        'T_arr_idx': 0,
        'T_dep_idx': 24,   # e.g., 6 hours (24 slots)
        'cdeg': 0.02
    }
    evs = [ev2, ev3]

    # GA parameters and objective weights (example)
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    alpha1, alpha2, alpha3 = 1000.0, 1000.0, 1000.0  # heavy penalties (tune as needed)

    # Run GA (small population / generations for demo)
    result = run_ga(evs, admitted_ids=None,
                    T=T, delta_t=delta_t,
                    pi_buy=pi_buy, pi_rev=pi_rev,
                    P_max=25.0,
                    weights=weights,
                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                    pop_size=40, ngen=80,
                    cxpb=0.9, mutpb=0.2,
                    eta_c=20.0, eta_m=20.0,
                    tournament_size=3,
                    stagnation_generations=20,
                    seed_count=4,
                    verbose=True)

    # Print readable schedule (M x T) for the best individual
    best_schedule = result['best_schedule']
    df = pd.DataFrame(best_schedule, index=[ev['id'] for ev in evs])
    print("\nBest schedule (rows: EV id, columns: slot index):")
    print(df)
    print("\nObjective breakdown:")
    print("J =", result['J'])
    print("F1 (net cost) =", result['F1'])
    print("F2 (degradation) =", result['F2'])
    print("F3 (variance) =", result['F3'])
    print("F4 (satisfaction sum) =", result['F4'])
    print("Violations (SoC, occ, grid):", result['V_SoC'], result['V_occ'], result['V_grid'])
