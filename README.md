# MIP-only demo — Internal Logistics PD-VRP

This folder is a self-contained, MIP-only version of the Internal Logistics PD-VRP
project. It includes the MIP formulation, configuration loader, result writer,
solution verifier, and visualization — but **none of the Solomon/NSGA-II
heuristics or test sweeps**. Use it to study the MIP model in isolation.

## What's inside

```
mip_only/
├── run_model.py              # MIP build + solve + result writer + multi-obj sweep
├── verify.py                 # Solution validator (auto-verify after each solve)
├── visualize.py              # Gantt / route / wait / route-timings PNGs
├── inputs/
│   ├── config.xlsx           # Simplified config (MIP keys only)
│   ├── nodes.xlsx            # Node list (depot + work centres)
│   ├── vehicles.xlsx         # Fleet definition (id, capacity, max_route, type, active)
│   ├── products.xlsx         # Four sheets: case1, case2, case3, case4
│   └── distances_minutes.xlsx  # OD travel-time matrix (long format)
└── results/                  # Created on first run
```

## Run modes

The config supports exactly two run modes:

- **`single_objective`** — one MIP solve. Optimises `primary_obj`
  (`route_duration` or `wait_time`) subject to `limit_on_constraint_obj` as an
  upper bound on the other objective.
- **`multi_objective`** — adaptive epsilon-constraint sweep over the Pareto
  front of `(route_duration, wait_time)`. Tightens the epsilon after each
  feasible solve; stops when infeasible.

The `objective_method` parameter chooses between Gurobi's `lexicographic`
multi-objective treatment and the classical `augmented_eps` (weighted-sum
augmentation).

## Quick start

```bash
cd mip_only

# 1. (Optional) Edit inputs/config.xlsx to pick a case, n, and run mode.
# 2. Run the model:
python run_model.py --inputs inputs --config inputs/config.xlsx
```

Outputs land in `results/run_<case>_n<NN>_<timestamp>/`:

- `result.xlsx` — summary sheet + variable sheets (x_used, assignment_f,
  itinerary, wait_w, route_timings, …)
- `route.png`, `gantt.png`, `waits.png`, `route_timings.png` — auto-generated
  visualisations (when `auto_visualize=True`)
- `gurobi.log` — Gurobi solver log

## Important configuration knobs

Edit `inputs/config.xlsx`:

| Parameter | Purpose |
|---|---|
| `run_mode` | `single_objective` or `multi_objective` |
| `primary_obj` | `route_duration` or `wait_time` |
| `constraint_obj` | the other objective |
| `product_set_id` | `case1`, `case2`, `case3`, or `case4` |
| `num_products` | integer slice of the product sheet, or `all` |
| `shift_duration_min` | T_max in shift-relative minutes (480 = 8 h) |
| `time_limit_seconds` | Gurobi wall-clock cap |
| `mip_gap` | Gurobi relative optimality tolerance |
| `objective_method` | `lexicographic` or `augmented_eps` |
| `limit_on_constraint_obj` | upper bound on secondary objective (single_objective mode) |
| `eps_step` | epsilon decrement between multi_objective iterations (in minutes) |

Valid-inequality / LP-tightening flags (`add_*_cut`, `tight_*`, `mtz_type`,
`break_vehicle_symmetry`, `use_indicator_constraints`) are all on by default.
Turn them off one at a time to see how each affects the LP relaxation and
solve time.

## Test cases

`products.xlsx` ships with four sheets:

- **case1** — distinct origin/destination pairs (basic).
- **case2** — concentrated locations (shared nodes).
- **case3** — shared-pickup cluster (tight precedence).
- **case4** — larger 100-product set (use small `num_products` slices).

## Requirements

- Python 3.10+
- `gurobipy` (with a valid Gurobi license)
- `pandas`, `openpyxl`, `matplotlib`, `numpy`

## Notes for the student

- Read `verify.py` to understand the solution-correctness contract — every
  MIP solve is verified against the original equations.
- `visualize.py` draws routes on a synthetic node layout (the original
  factory floor plan is not bundled). Modify `NODE_POSITIONS` there if you
  want a custom layout.
- The MIP formulation, big-M choices, and valid inequalities in
  `run_model.py:build_model` match the LaTeX writeup of the model. The
  function is byte-identical to the main-project version — the only thing
  removed from this demo is the heuristic/NSGA-II/test machinery.
