# Student version — Internal Logistics PD-VRP

Self-contained version of the Internal Logistics PD-VRP project for teaching
and student work. It includes:

- The full MIP formulation (single-objective and epsilon-constraint multi-objective)
- The Solomon I1 construction heuristic, with 2-opt and Or-opt-on-pairs local
  search, plus the full repair ladder (bounded-depth backtracking + multi-start
  over alpha-biases + extended biases + random alpha-restarts + shuffled-order
  restarts).

It excludes NSGA-II and all test-sweep modes that we used internally for
experiments.

## What's inside

```
student_version/
├── run_model.py              # MIP + heuristic dispatch + result writers
├── solomon.py                # Solomon I1 + 2-opt + Or-opt + repair ladder
├── verify.py                 # Solution validator
├── visualize.py              # Gantt / route / wait / route-timings PNGs
├── README.md                 # This file
└── inputs/
    ├── config.xlsx           # Simplified config (MIP + Solomon keys only)
    ├── nodes.xlsx            # Node list (depot + work centres)
    ├── vehicles.xlsx         # Fleet definition (id, capacity, max_route, type, active)
    ├── products.xlsx         # Four sheets: case1, case2, case3, case4
    └── distances_minutes.xlsx  # OD travel-time matrix (long format)
```

## Run modes

Three run modes are exposed through `inputs/config.xlsx`:

| `run_mode`         | What it does |
|--------------------|--------------|
| `single_objective` | One MIP solve. Optimises `primary_obj` under `limit_on_constraint_obj`. |
| `multi_objective`  | Adaptive epsilon-constraint sweep over the Pareto front of `(route_duration, wait_time)`. |
| `heuristic`        | Solomon I1 construction + 2-opt + Or-opt; no MIP. Solves much faster than MIP, gives a feasible (often near-optimal) solution. |

The `objective_method` parameter selects between Gurobi's `lexicographic`
multi-objective treatment and the classical `augmented_eps` (weighted-sum
augmentation).

## Quick start

```bash
cd student_version

# 1. (Optional) Edit inputs/config.xlsx to pick a case, n, and run_mode.
# 2. Run:
python run_model.py --inputs inputs --config inputs/config.xlsx
```

Outputs land in `results/<label>_<timestamp>/`:

- `result_*.xlsx` — summary sheet + variable sheets (x_used, assignment_f,
  itinerary, wait_w, route_timings, …)
- `route.png`, `gantt.png`, `waits.png`, `route_timings.png` — auto-generated
  visualisations (when `auto_visualize=True`)
- `gurobi.log` — Gurobi solver log (single_objective and multi_objective only)

## Key configuration knobs

### MIP

| Parameter | Purpose |
|---|---|
| `run_mode` | `single_objective`, `multi_objective`, or `heuristic` |
| `primary_obj` / `constraint_obj` | `route_duration` or `wait_time` |
| `objective_method` | `lexicographic` or `augmented_eps` |
| `limit_on_constraint_obj` | upper bound on secondary objective (single_objective mode) |
| `eps_step` | epsilon decrement between multi_objective iterations (minutes) |
| `product_set_id` | `case1`, `case2`, `case3`, or `case4` |
| `num_products` | integer slice, or `all` |
| `shift_duration_min` | T_max in shift-relative minutes (480 = 8 h) |
| `time_limit_seconds` | Gurobi wall-clock cap |
| `mip_gap` | Gurobi relative optimality tolerance |

Valid-inequality / LP-tightening flags (`add_*_cut`, `tight_*`, `mtz_type`,
`break_vehicle_symmetry`, `use_indicator_constraints`) are all on by default.
Toggle them one at a time to see how each affects LP relaxation and solve
time.

### Solomon heuristic

| Parameter | Purpose |
|---|---|
| `heuristic_objective` | Bias label (`route_duration` or `wait_time`). |
| `alpha_1`, `alpha_2`, `alpha_3` | Weights in `c_1 = α₁·Δd + α₂·ΔT + α₃·ΔW`. Defaults are the strict-lex triple `(0.0001, 1, 0.01)` for the route-duration bias. |
| `lambda_c2` | Scaling on `c(h, o_p)` in `c_2`. |
| `heuristic_wait_limit` | Epsilon bound on total wait. |
| `apply_2opt`, `apply_or_opt` | Toggle the two local-search phases. |
| `solomon_backtrack_max_depth` | Bounded-depth backtracking when greedy reaches `infeasible_unrouted`. n=20 instances typically need depth ≥ 15. |
| `solomon_multistart` | After backtracking fails, retry under the other stock biases (route_duration → wait_time → balanced → distance → distance+wait → distance+duration). |
| `solomon_random_restarts` | After the 6 stock biases fail, try this many log-uniform random alpha triples. |
| `solomon_shuffle_restarts` | After random alpha-restarts fail, try this many shuffled-order restarts (random insertion order, no c2 ranking, no backtracking). Defeats tight single-route instances. |
| `solomon_random_seed` | Seed for reproducible random/shuffle restarts. |

### Test cases

`products.xlsx` ships with four sheets:

- **case1** — distinct origin/destination pairs (basic).
- **case2** — concentrated locations (shared nodes).
- **case3** — shared-pickup cluster (tight precedence).
- **case4** — larger 100-product set (use small `num_products` slices).

## Requirements

- Python 3.10+
- `gurobipy` (with a valid Gurobi license, for `single_objective` and `multi_objective`)
- `pandas`, `openpyxl`, `matplotlib`, `numpy`

The Solomon heuristic does **not** need Gurobi — run it with
`run_mode = heuristic` to explore the problem without a Gurobi license.

## Notes for the student

- Read `verify.py` to understand the solution-correctness contract. Every
  MIP and every Solomon solve is validated against the original
  equations, so a "feasible" report has been independently checked.
- `visualize.py` draws routes on a synthetic node layout (the original
  factory floor plan is not bundled). Modify `NODE_POSITIONS` there if you
  want a custom layout.
- The MIP formulation, big-M choices, valid inequalities, and Solomon
  repair logic in this folder are byte-identical to the versions in the
  research project — the only thing removed for this student version is
  NSGA-II and the test-sweep machinery.
