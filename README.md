````markdown
# Optimal EV Charging Scheduler with Priority Admission and Multi-Objective Optimization

**Status:** Prototype / research code (Python + some supporting C++ sample code).

## Project Overview

This project implements a two-stage EV charging scheduler suitable for a depot/parking site where chargers are limited.

* **Stage I — Admission:** Chooses which EVs are admitted to the limited chargers (M) using either a priority ranking (based on urgency, battery degradation risk, grid stress, price sensitivity, etc.) or a baseline First-Come-First-Served (FCFS) policy.
* **Stage II — Scheduling:** For the admitted EVs, a Genetic Algorithm (GA) produces a per-timeslot charging plan that minimizes energy cost and battery degradation, reduces load variance on the grid, maximizes user satisfaction, and enforces SoC / charger / grid constraints.

## Features

-   Two-stage pipeline: admission (priority / FCFS) + GA-based scheduling.
-   Multi-objective optimization (cost, degradation, load smoothing, satisfaction).
-   Synthetic EV generator for experiments.
-   Scripts to run the pipeline and evaluate resulting schedules (CSV outputs included).

## Getting Started

### Prerequisites

-   Python 3.8+ (recommended)
-   Standard Python tooling: `pip`, `venv` (or `conda`)

### Install

1.  Clone the repository:
    ```bash
    git clone [https://github.com/codexankitsingh/Optimal_cost_reduction_in_electric_vehicle_Using_Priority_scheduling_and_Multi_objective_optimisatio.git](https://github.com/codexankitsingh/Optimal_cost_reduction_in_electric_vehicle_Using_Priority_scheduling_and_Multi_objective_optimisatio.git)
    cd Optimal_cost_reduction_in_electric_vehicle_Using_Priority_scheduling_and_Multi_objective_optimisatio
    ```

2.  (Optional) Create and activate a virtual environment:
    ```bash
    # macOS / Linux
    python -m venv .venv
    source .venv/bin/activate
    
    # Windows (PowerShell/CMD)
    # .venv\Scripts\activate
    ```

3.  Install Python dependencies (if `requirements.txt` is present):
    ```bash
    pip install -r requirements.txt
    ```
    *(If the repository does not include `requirements.txt`, install commonly used packages like `numpy`, `pandas`, and `matplotlib` as needed.)*

### Run Examples

The repo provides a pipeline script and a GA script. Adjust file names and arguments to match the files in your local clone.

**Example 1 — Run the full pipeline:**
```bash
python pipeline_full.py --ev-file evs.csv --M 30 --T 48 --mode ga_priority
````

**Example 2 — Run GA stage directly:**

```bash
python ga.py --input admitted_evs.csv --T 48 --generations 200 --pop-size 100
```

**Example 3 — Alternative entry point (if available):**

```bash
# If you have a main.py wrapper (some forks/versions include this)
python main.py --ev-file evs.csv --M 30 --T 48 --mode ga_priority
```

If any script expects different argument names, check its help message:

```bash
python <script_name>.py --help
```

*(Commands above are adapted from the repository examples and file listing.)*

## Repository Structure (Brief)

```
.
├─ .vscode/                       # Editor settings (optional)
├─ README.md                      # This file
├─ evs.csv                        # Sample/generated EV dataset
├─ ga.py                          # Genetic Algorithm scheduling implementation
├─ pipeline_full.py               # Full admission + scheduling pipeline
├─ priority_scheduling.py         # Priority / admission policies (Python)
├─ priority_sch.cpp               # C++ priority scheduler (sample)
├─ k.cpp                          # Additional C++ helper(s)
├─ best_schedule_improved.csv     # Example results
├─ best_schedule_normalized.csv   # Example results (normalized)
├─ pipeline_metrics_summary.csv   # Metrics summary produced by pipeline
└─ ...
```

*Files and names are taken from the repository listing — inspect each script to confirm exact argument names and behavior.*

## Algorithms & Approach

### Stage I: Admission

  - **Priority-based admission:** EVs are scored (e.g., based on remaining SoC, required departure time, battery degradation sensitivity, price sensitivity). Top-M EVs are admitted.
  - **FCFS baseline:** EVs are admitted by arrival order.

### Stage II: Scheduling (Genetic Algorithm)

  - **Encoding:** A chromosome describes the power allocation across all time slots for each admitted EV.
  - **Objectives:** Minimize total energy cost, minimize estimated battery degradation, minimize the variance of the total load (grid smoothing), and maximize user satisfaction (SoC targets met by deadlines).
  - **Genetic Operators:** Population initialization, selection (tournament / roulette), crossover, mutation, and constraint repair to keep schedules feasible.
  - **Output:** The best per-timeslot schedule (CSV) and a summary of performance metrics.

## Input / Datasets

The primary input is a CSV file (e.g., `evs.csv`) containing details for each EV. Typical columns include:

  - `ev_id` — unique identifier
  - `arrival_time` — time slot index when vehicle arrives
  - `departure_time` — last allowed time slot to finish charging
  - `soc_initial` — initial state-of-charge (%) or kWh
  - `soc_target` — required SoC at departure (%) or kWh
  - `capacity_kwh` — battery capacity (kWh)
  - `max_charge_kw` — max charging power (kW)
  - `priority / urgency` — (optional) precomputed priority score or flags

*Open the sample `evs.csv` to confirm exact column names and data format.*

## Outputs / Results

  - `best_schedule_improved.csv` / `best_schedule_normalized.csv` — Example schedules produced by the GA, showing the power allocated to each EV at each time slot.
  - `pipeline_metrics_summary.csv` — A summary file generated after each run, containing key performance indicators (total cost, average degradation, load variance, % of SoC targets met, etc.).

Use these CSVs for plotting and comparing the performance of different admission policies and GA configurations.

## Tips for Experimentation

  - **Start Small:** Begin with a small number of EVs (e.g., 20–50) and a short time horizon (T = 24–48) to iterate quickly.
  - **Tune GA Hyperparameters:** Experiment with population size, number of generations, and crossover & mutation rates. Monitor convergence and runtime.
  - **Compare Policies:** Run the pipeline with different admission policies (`priority` vs. `FCFS`) and compare the resulting metrics in `pipeline_metrics_summary.csv`.
  - **Optimize Performance:** If runtime is slow, consider parallelizing the fitness evaluation or using a smaller initial population.

## Contributing

1.  Fork the repo.
2.  Create a feature branch: `git checkout -b feat/my-improvement`.
3.  Add tests or example runs demonstrating improvements.
4.  Open a Pull Request describing the change and its impact on metrics.

## License & Contact

Include a license file in the repo (e.g., MIT) if you want to open-source this code publicly.

  * **Author:** Ankit Kumar Singh

<!-- end list -->

```
```
