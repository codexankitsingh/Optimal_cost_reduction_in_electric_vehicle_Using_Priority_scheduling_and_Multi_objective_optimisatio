# ⚡ EV Charging Scheduler — README

**Project:** EV Charging Scheduler (Two-stage: Admission + Scheduling)  
**Author:** Ankit Kumar Singh  
**Language:** Python  
**Status:** Prototype / Research code (GA-based Stage-II, Priority & FCFS Stage-I)

---

## Summary

This repository implements a two-stage scheduling framework for electric vehicle (EV) charging at a depot/parking site:

- **Stage-I — Admission**: Select which EVs get access to limited chargers (M chargers) using:
  - Priority ranking (urgency, battery degradation, grid-stress, price);
  - or FCFS (arrival order) as baseline.

- **Stage-II — Scheduling**: For the admitted EVs, a Genetic Algorithm (GA) computes a per-timeslot charging schedule to:
  - minimize total energy cost and battery degradation,
  - smooth grid load (reduce variance),
  - maximize user satisfaction,
  - satisfy constraints (SoC targets, charger occupancy, grid capacity).

---

## Quickstart

### Installation
```bash
pip install -r requirements.txt


## Run example
python main.py --ev-file evs.csv --M 30 --T 48 --mode ga_priority

