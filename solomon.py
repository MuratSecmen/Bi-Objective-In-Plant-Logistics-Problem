"""
solomon.py — Solomon I1-based sequential insertion heuristic for the
Internal Logistics PD-VRP.

The heuristic consumes the same `Instance` object as the MIP (built by
`load_instance` in run_model.py) and produces a `Solution` object
(defined in verify.py) compatible with every downstream tool that the
MIP path uses: result.xlsx writers, the verifier, the five PNG
visualisations.

Design notes
============

Naming conventions match the MIP code exactly:
    inst.K               vehicles
    inst.routes_of(k)    routes available to vehicle k, 1..max_routes[k]
    inst.N, inst.Nw      active nodes (depot + work stations)
    inst.P               products
    inst.o[p], inst.d[p] origin / destination of product p
    inst.e[p]            ready time (shift-relative)
    inst.s_load[p]       loading service time
    inst.s_unload[p]     unloading service time
    inst.q_p[p]          product surface area
    inst.q_k[k]          vehicle capacity
    inst.c[(i, j)]       travel time from i to j
    inst.T_max           shift duration

Algorithm
=========

The heuristic is a multi-route generalisation of Solomon I1 sequential
insertion. The fleet has |KR_pairs| empty route slots from the start
(one per valid (k, r) pair, drawn from vehicles.xlsx's max_route
column). At each iteration:

  1. For every unrouted product p, scan every (k, r) slot and every
     feasible insertion position; record the cheapest c_1 over all
     (k, r, position) triples. Cases A / B / C / D handle whether
     o_p and/or d_p already appear in route (k, r).
  2. Among parts with at least one feasible insertion across the
     fleet, pick the one with the highest selection score
        c_2(p) = lambda * c(h, o_p) - c_1^*(p).
  3. Commit the chosen insertion. Subtract p from the unrouted set.
  4. Repeat until either the unrouted set is empty (full feasibility)
     or no remaining part has a feasible insertion (infeasible).

Per-vehicle route chaining
--------------------------

Constraint (14) of the MIP forces route r+1 to start no earlier than
route r returns. The simulator honours this: for vehicle k it computes
route 1 first starting at T_start = 0, then route 2 starting at the
depot return time of route 1, and so on. Vehicles are independent.

Wait-time epsilon constraint
----------------------------

Total waiting time across all products is bounded above by
`heuristic_wait_limit` (config-driven). Insertions that would push
the total wait above the limit are rejected.

Insertion cost
--------------

c_1 = alpha_1 * Δd
    + alpha_2 * max(0, Δ fleet-wide route duration)
    + alpha_3 * max(0, Δ fleet-wide total wait)

Δd is the arc-distance increase from the trial insertion.

The two-objective bias is steered through (alpha_1, alpha_2, alpha_3):
  - Minimise route duration: alpha_1 = 1, alpha_2 = 1, alpha_3 = 0
  - Minimise total wait:     alpha_1 = 1, alpha_2 = 0, alpha_3 = 1
  - Balanced:                alpha_1 = 1, alpha_2 = 0.5, alpha_3 = 0.5

`heuristic_objective` is a label-only config flag that documents the
chosen bias (used for reporting and future 2-opt local search).
"""

from __future__ import annotations

import copy
import logging
import math
import time as _time
from dataclasses import dataclass, field
from typing import Optional

from verify import Solution

log = logging.getLogger("internal_logistics.solomon")

EPS = 1e-9


# =============================================================================
# Per-route state
# =============================================================================
@dataclass
class RouteState:
    """One (vehicle, route) slot under construction."""
    k: str
    r: int
    # Ordered node sequence: starts and ends at depot 'h'.
    # Initially [h, h] meaning empty route.
    nodes: list = field(default_factory=lambda: ["h", "h"])
    # Products assigned to this route.
    parts: set = field(default_factory=set)

    def is_empty(self) -> bool:
        return len(self.parts) == 0

    def customer_nodes(self) -> list:
        return self.nodes[1:-1]

    def node_position(self, node) -> int:
        """Return the index of `node` in self.nodes, or -1 if absent.

        Searches only customer positions (excluding the two depot anchors).
        """
        for idx in range(1, len(self.nodes) - 1):
            if self.nodes[idx] == node:
                return idx
        return -1

    def deepcopy(self) -> "RouteState":
        clone = RouteState(k=self.k, r=self.r)
        clone.nodes = list(self.nodes)
        clone.parts = set(self.parts)
        return clone


# =============================================================================
# Per-route simulation
# =============================================================================
def _route_rt_origin(route_parts: set, node, inst) -> float:
    """Lower bound on service-start time at `node` due to ready times.

    Only the parts CURRENTLY ASSIGNED to this route contribute. This is
    the per-route analogue of MIP constraint (18):
        t^s_{o_p, k, r}  >=  e_p   when f_{pkr} = 1.
    A part not on the route imposes no constraint at its origin.
    """
    return max((inst.e[p] for p in route_parts if inst.o[p] == node),
               default=0.0)


def simulate_route(route: RouteState, inst, T_start: float = 0.0) -> dict:
    """Walk `route.nodes` forward from depot, computing ta, td, ts, y.

    Returns a dict with keys:
        feasible       : bool
        reason         : str, only when not feasible
        ta             : {(node, k, r): arrival time}
        td             : {(node, k, r): departure time}
        ts             : {(node, k, r): service-start time}, only for Nw
        y              : {(node, k, r): load on board AFTER node}
        depot_arrival  : float, time the truck returns to h
        waits          : {p: w_p} for parts on this route
        arrivals_to_d  : {p: t^a at d_p} for parts on this route
    """
    k, r = route.k, route.r

    # Empty route — truck never departs the depot. Depot arrival equals
    # depot departure equals T_start. Matches MIP eq (4)/(5) when the
    # route is unused (Σ x_{hj} = 0): ta[h, k, r] just floats up to
    # ta[h, k, r-1] via constraint (15), which equals T_start here.
    if not route.customer_nodes():
        return {
            "feasible": True,
            "ta": {("h", k, r): T_start},
            "td": {("h", k, r): T_start},
            "ts": {},
            "y": {("h", k, r): 0.0},
            "depot_arrival": T_start,
            "arrivals_to_d": {},
            "waits": {},
        }

    prev = "h"
    td_prev = T_start
    load = 0.0
    cap = inst.q_k[k]

    ta = {("h", k, r): T_start}
    td = {("h", k, r): T_start}
    ts: dict = {}
    y = {("h", k, r): 0.0}

    # Group parts by their pickup / delivery node on THIS route.
    cust = route.customer_nodes()
    pickups_at = {n: [] for n in cust}
    deliveries_at = {n: [] for n in cust}
    for p in route.parts:
        op, dp = inst.o[p], inst.d[p]
        if op in pickups_at:
            pickups_at[op].append(p)
        if dp in deliveries_at:
            deliveries_at[dp].append(p)

    arrivals_to_d: dict = {}

    for n in cust:
        # Arrival at n via the traversed arc (MIP constraint 16 chained).
        ta_n = td_prev + inst.c.get((prev, n), float("inf"))
        if not math.isfinite(ta_n):
            return {"feasible": False, "reason": f"missing_arc[{prev}->{n}]"}

        # Service-start time at n:
        #   - must wait for unload of any deliveries arriving here (eq 17)
        #   - must wait for ready time of any pickups here (eq 18)
        unload = sum(inst.s_unload[p] for p in deliveries_at[n])
        load_time = sum(inst.s_load[p] for p in pickups_at[n])
        ts_n = max(ta_n + unload, _route_rt_origin(route.parts, n, inst))

        # Departure (eq 19).
        td_n = ts_n + load_time

        # Capacity tracking (eq 23-26 along the route).
        load -= sum(inst.q_p[p] for p in deliveries_at[n])
        load += sum(inst.q_p[p] for p in pickups_at[n])
        if load > cap + EPS:
            return {
                "feasible": False,
                "reason": f"capacity_overflow[{n}]: load={load:.3f} > cap={cap}",
            }

        # Record arrival at destinations for wait-time calculation.
        for p in deliveries_at[n]:
            arrivals_to_d[p] = ta_n

        ta[(n, k, r)] = ta_n
        td[(n, k, r)] = td_n
        ts[(n, k, r)] = ts_n
        y[(n, k, r)] = load

        prev = n
        td_prev = td_n

    # Final hop back to depot.
    depot_arrival = td_prev + inst.c.get((prev, "h"), float("inf"))
    if not math.isfinite(depot_arrival):
        return {"feasible": False, "reason": f"missing_arc[{prev}->h]"}

    # Shift-window check (relative to T_start: depot return must fit T_max).
    if depot_arrival > inst.T_max + EPS:
        return {
            "feasible": False,
            "reason": (
                f"shift_overflow: depot_arrival={depot_arrival:.3f}"
                f" > T_max={inst.T_max}"
            ),
        }

    # ta[("h", k, r)] for the LAST visit to h is the return time, which
    # is what the MIP objective uses. We overwrite the depot-departure
    # entry stored at T_start with the return time.
    ta[("h", k, r)] = depot_arrival

    # Pickup-before-delivery precedence (MIP eq 20 and eq 32 satisfied
    # automatically by the inserted-order convention, but verify).
    for p in route.parts:
        op, dp = inst.o[p], inst.d[p]
        if (op, k, r) not in td or (dp, k, r) not in ta:
            return {"feasible": False, "reason": f"missing_visit[{p}]"}
        if ta[(dp, k, r)] + EPS < td[(op, k, r)]:
            return {
                "feasible": False,
                "reason": f"precedence_violation[{p}]: ta(d)={ta[(dp,k,r)]:.3f} < td(o)={td[(op,k,r)]:.3f}",
            }

    # Per-product wait time (MIP eq 22 at integer optimum).
    waits = {}
    for p in route.parts:
        ta_dp = arrivals_to_d[p]
        waits[p] = max(0.0, ta_dp + inst.s_unload[p] - inst.e[p])

    return {
        "feasible": True,
        "ta": ta, "td": td, "ts": ts, "y": y,
        "depot_arrival": depot_arrival,
        "arrivals_to_d": arrivals_to_d,
        "waits": waits,
    }


# =============================================================================
# Per-vehicle chained simulation
# =============================================================================
def simulate_vehicle(routes_of_k: list, inst) -> list:
    """Simulate all routes of one vehicle in order, chaining start times.

    Returns a list of sim dicts (one per route). Route r's T_start is
    the depot return time of route r-1 (MIP eq 14). If any route is
    infeasible, all subsequent routes' sims are marked infeasible too.
    """
    sims = []
    T_current = 0.0
    failed = False
    for route in routes_of_k:
        if failed or not math.isfinite(T_current):
            sims.append({"feasible": False, "reason": "prior_route_failed"})
            continue
        # If the prior route already used T_max, any new route would overflow.
        if T_current > inst.T_max + EPS:
            sims.append({"feasible": False, "reason": "prior_route_used_all_time"})
            failed = True
            continue
        sim = simulate_route(route, inst, T_start=T_current)
        sims.append(sim)
        if sim["feasible"]:
            T_current = sim["depot_arrival"]
        else:
            failed = True
    return sims


# =============================================================================
# Fleet state — collection of all (k, r) routes
# =============================================================================
@dataclass
class FleetState:
    inst: object
    # routes keyed by (k, r); created for every valid pair from inst.KR_pairs.
    routes: dict = field(default_factory=dict)
    # Cached sims keyed by k; invalidated whenever a route on vehicle k changes.
    _sim_cache: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, inst) -> "FleetState":
        state = cls(inst=inst)
        for (k, r) in inst.KR_pairs:
            state.routes[(k, r)] = RouteState(k=k, r=r)
        return state

    def routes_of_vehicle(self, k) -> list:
        return [self.routes[(k, r)] for r in self.inst.routes_of(k)]

    def invalidate(self, k):
        self._sim_cache.pop(k, None)

    def simulate_vehicle(self, k) -> list:
        if k not in self._sim_cache:
            self._sim_cache[k] = simulate_vehicle(
                self.routes_of_vehicle(k), self.inst
            )
        return self._sim_cache[k]

    def total_route_duration(self) -> float:
        """Sum over vehicles of the depot-return time on the LAST route.

        Mirrors the MIP objective f1 = Σ_k t^a[h, k, |R_k|].
        """
        total = 0.0
        for k in self.inst.K:
            sims = self.simulate_vehicle(k)
            # Use the highest-r route's depot_arrival; if that route is
            # empty it just stays at the chain-start time, which is the
            # return time of the last NON-empty route.
            last_sim = sims[-1]
            if last_sim["feasible"]:
                total += last_sim["depot_arrival"]
            else:
                return float("inf")
        return total

    def total_wait(self) -> float:
        """Sum of per-product wait times across all routes."""
        total = 0.0
        for k in self.inst.K:
            sims = self.simulate_vehicle(k)
            for sim in sims:
                if sim["feasible"]:
                    total += sum(sim["waits"].values())
        return total

    def unrouted(self, all_parts) -> set:
        assigned = set()
        for route in self.routes.values():
            assigned |= route.parts
        return all_parts - assigned


def _clone_fleet(fleet: "FleetState") -> "FleetState":
    """Shallow-but-correct deep clone of a fleet.

    Copies every route (which deep-copies its node list and parts set), shares
    the read-only ``inst`` reference, and starts with an empty sim cache so the
    clone is safe to mutate independently of the original.
    """
    clone = FleetState(inst=fleet.inst)
    clone.routes = {
        key: route.deepcopy() for key, route in fleet.routes.items()
    }
    return clone


# =============================================================================
# Backtracking — decision snapshots
# =============================================================================
@dataclass
class _Decision:
    """One Solomon insertion decision, with enough state to be undone.

    A decision records the *ranked* list of candidate (product, plan) tuples
    that were available at this iteration, which alternative was actually
    applied (``chosen_idx``), and a snapshot of the fleet/unrouted set just
    BEFORE the decision was applied. The snapshot lets the bounded-depth
    backtracker rewind to this point and try a different alternative.
    ``fleet_before`` is ``None`` when snapshots are disabled (e.g. while the
    backtracker is itself completing a candidate trial — we do not nest
    backtracking inside backtracking).
    """
    iteration: int
    fleet_before: object         # FleetState clone, or None
    unrouted_before: object      # set[str], or None
    alternatives: list           # [(p, InsertionPlan, c2), ...] sorted as Solomon ranks
    chosen_idx: int = 0


# =============================================================================
# Insertion logic
# =============================================================================
@dataclass
class InsertionPlan:
    k: str
    r: int
    case: str           # "A", "B", "C", "D"
    pos_o: Optional[int] = None
    pos_d: Optional[int] = None
    c1: float = float("inf")


def _delta_distance_two(route: RouteState, i: int, j: int,
                         o: str, d: str, inst) -> float:
    """Arc-distance increase from inserting o at i and d at j (case A)."""
    r = route.nodes
    d_o = (inst.c.get((r[i - 1], o), 0.0)
           + inst.c.get((o, r[i]), 0.0)
           - inst.c.get((r[i - 1], r[i]), 0.0))
    # After inserting o at i, the node list shifts; compute d's delta
    # against the shifted sequence.
    tmp = r[:i] + [o] + r[i:]
    d_d = (inst.c.get((tmp[j - 1], d), 0.0)
           + inst.c.get((d, tmp[j]), 0.0)
           - inst.c.get((tmp[j - 1], tmp[j]), 0.0))
    return d_o + d_d


def _delta_distance_one(route: RouteState, pos: int,
                         node: str, inst) -> float:
    """Arc-distance increase from inserting `node` at position `pos`."""
    r = route.nodes
    return (inst.c.get((r[pos - 1], node), 0.0)
            + inst.c.get((node, r[pos]), 0.0)
            - inst.c.get((r[pos - 1], r[pos]), 0.0))


def best_insert_into_route(p, route: RouteState, fleet: FleetState,
                            cfg: dict, baseline_dur: float,
                            baseline_wait: float) -> InsertionPlan:
    """Find the cheapest feasible insertion of part p into one route.

    Tries cases A, B, C, D as appropriate. Returns an InsertionPlan with
    c1 = +inf if no feasible insertion exists.
    """
    inst = fleet.inst
    k, r = route.k, route.r
    op, dp = inst.o[p], inst.d[p]
    o_pos = route.node_position(op)
    d_pos = route.node_position(dp)

    alpha1 = cfg["alpha_1"]
    alpha2 = cfg["alpha_2"]
    alpha3 = cfg.get("alpha_3", 0.0)
    wait_limit = cfg["heuristic_wait_limit"]

    best = InsertionPlan(k=k, r=r, case="")

    def _try(case: str, trial: RouteState, dist_increase: float,
             pos_o: Optional[int], pos_d: Optional[int]):
        nonlocal best
        # Simulate the affected vehicle with the trial route swapped in.
        original = fleet.routes[(k, r)]
        fleet.routes[(k, r)] = trial
        fleet.invalidate(k)
        try:
            sims = fleet.simulate_vehicle(k)
            if not all(s["feasible"] for s in sims):
                return
            new_dur = fleet.total_route_duration()
            new_wait = fleet.total_wait()
            if new_wait > wait_limit + EPS:
                return
            c1 = (alpha1 * dist_increase
                  + alpha2 * max(0.0, new_dur - baseline_dur)
                  + alpha3 * max(0.0, new_wait - baseline_wait))
            if c1 < best.c1:
                best = InsertionPlan(
                    k=k, r=r, case=case,
                    pos_o=pos_o, pos_d=pos_d, c1=c1,
                )
        finally:
            fleet.routes[(k, r)] = original
            fleet.invalidate(k)

    # Case D: both nodes already in the route; just add the assignment.
    if o_pos >= 0 and d_pos >= 0:
        if o_pos < d_pos:
            trial = route.deepcopy()
            trial.parts.add(p)
            _try("D", trial, dist_increase=0.0, pos_o=None, pos_d=None)
        return best

    # Case A: neither node present.
    if o_pos < 0 and d_pos < 0:
        n = len(route.nodes)
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                trial = route.deepcopy()
                trial.nodes.insert(i, op)
                trial.nodes.insert(j, dp)
                trial.parts.add(p)
                dd = _delta_distance_two(route, i, j, op, dp, inst)
                _try("A", trial, dd, i, j)
        return best

    # Case B: o_p present, d_p absent.
    # j is the position where d_p will be inserted *into the original* route.
    # Valid positions are {o_pos+1, ..., n-1}: j=n-1 places d_p just before
    # the closing depot, j=n would place d_p AFTER the closing depot
    # (producing a malformed route [..., h, d_p]) and _delta_distance_one
    # would then dereference r[n] -> IndexError. Upper bound is therefore n,
    # exclusive.
    if o_pos >= 0 and d_pos < 0:
        n = len(route.nodes)
        for j in range(o_pos + 1, n):
            trial = route.deepcopy()
            trial.nodes.insert(j, dp)
            trial.parts.add(p)
            dd = _delta_distance_one(route, j, dp, inst)
            _try("B", trial, dd, None, j)
        return best

    # Case C: d_p present, o_p absent.
    if o_pos < 0 and d_pos >= 0:
        for i in range(1, d_pos + 1):
            trial = route.deepcopy()
            trial.nodes.insert(i, op)
            trial.parts.add(p)
            dd = _delta_distance_one(route, i, op, inst)
            _try("C", trial, dd, i, None)
        return best

    return best


def best_insert_anywhere(p, fleet: FleetState, cfg: dict,
                          baseline_dur: float, baseline_wait: float
                          ) -> InsertionPlan:
    """Try inserting p into every (k, r) slot; return the cheapest plan."""
    best = InsertionPlan(k="", r=0, case="")
    for (k, r), route in fleet.routes.items():
        plan = best_insert_into_route(
            p, route, fleet, cfg, baseline_dur, baseline_wait
        )
        if plan.c1 < best.c1:
            best = plan
    return best


# =============================================================================
# Apply an accepted insertion
# =============================================================================
def apply_insertion(p, plan: InsertionPlan, fleet: FleetState):
    """Mutate the fleet to commit the chosen insertion."""
    route = fleet.routes[(plan.k, plan.r)]
    op = fleet.inst.o[p]
    dp = fleet.inst.d[p]
    route.parts.add(p)
    if plan.case == "A":
        route.nodes.insert(plan.pos_o, op)
        route.nodes.insert(plan.pos_d, dp)
    elif plan.case == "B":
        route.nodes.insert(plan.pos_d, dp)
    elif plan.case == "C":
        route.nodes.insert(plan.pos_o, op)
    elif plan.case == "D":
        pass  # nodes already present
    fleet.invalidate(plan.k)


# =============================================================================
# Main construction loop
# =============================================================================
def _greedy_insert_loop(inst, cfg, fleet, unrouted, decisions,
                         take_snapshots=True):
    """One greedy pass of Solomon I1 from the current ``fleet``/``unrouted`` state.

    Mutates ``fleet`` and ``unrouted`` in place and appends every accepted
    insertion to ``decisions``. Returns ``"feasible"`` if every part is routed
    or ``"infeasible_unrouted"`` if some iteration finds no feasible
    insertion for any remaining product.

    When ``take_snapshots`` is True, each appended ``_Decision`` carries a
    pre-decision fleet clone and unrouted-set copy so a later backtracker can
    rewind to that point. Snapshots are skipped while the backtracker itself
    is completing a trial — we never nest backtracking inside backtracking.
    """
    lam = cfg["lambda_c2"]
    iteration = decisions[-1].iteration if decisions else 0

    while unrouted:
        iteration += 1
        baseline_dur = fleet.total_route_duration()
        baseline_wait = fleet.total_wait()

        # Score every unrouted part.
        scored = []
        for p in sorted(unrouted):
            plan = best_insert_anywhere(
                p, fleet, cfg, baseline_dur, baseline_wait,
            )
            if math.isfinite(plan.c1):
                c2 = lam * inst.c.get(("h", inst.o[p]), 0.0) - plan.c1
                scored.append((p, plan, c2))

        if not scored:
            return "infeasible_unrouted"

        # Pick the highest c2; tie-break by lower c1, then by part id.
        scored.sort(key=lambda t: (-t[2], t[1].c1, t[0]))

        # Snapshot BEFORE applying so the backtracker can rewind here.
        if take_snapshots:
            fleet_snapshot = _clone_fleet(fleet)
            unrouted_snapshot = set(unrouted)
        else:
            fleet_snapshot = None
            unrouted_snapshot = None

        p_best, plan_best, c2_best = scored[0]
        apply_insertion(p_best, plan_best, fleet)
        unrouted.discard(p_best)
        decisions.append(_Decision(
            iteration=iteration,
            fleet_before=fleet_snapshot,
            unrouted_before=unrouted_snapshot,
            alternatives=scored,
            chosen_idx=0,
        ))
        log.info(
            "iter %d: inserted %s into (%s, R%d) case=%s  c1=%.3f  c2=%.3f",
            iteration, p_best, plan_best.k, plan_best.r,
            plan_best.case, plan_best.c1, c2_best,
        )

    return "feasible"


def _try_backtrack(inst, cfg, fleet, decisions, max_depth):
    """Bounded-depth, iterative-deepening backtracking after a Solomon failure.

    Tries to recover from an ``infeasible_unrouted`` greedy result by undoing
    the last ``depth`` insertion decisions for ``depth`` = 1, 2, ..., up to
    ``max_depth``. At each depth, the rewind point is the snapshot recorded
    BEFORE that decision; we then iterate through the remaining alternatives
    that the original greedy step did *not* pick (in c2-rank order). For each
    alternative we apply it and run :func:`_greedy_insert_loop` with snapshots
    disabled (no nested backtracking). The first alternative that completes
    the construction feasibly wins.

    Returns ``(fleet, decisions)`` on success, ``None`` on failure.
    """
    for depth in range(1, max_depth + 1):
        if depth > len(decisions):
            log.info(
                "  backtrack: only %d decision(s) exist; cannot undo %d levels",
                len(decisions), depth,
            )
            break

        target = decisions[-depth]
        if target.fleet_before is None:
            # Snapshots were disabled at this decision — nothing to rewind to.
            continue

        n_alts = len(target.alternatives)
        n_remaining = n_alts - target.chosen_idx - 1
        orig_p, orig_plan, orig_c2 = target.alternatives[target.chosen_idx]
        log.info(
            "  backtrack depth %d: rewinding to iter %d (original=%s, "
            "%d alternative(s) to try)",
            depth, target.iteration, orig_p, n_remaining,
        )

        for alt_idx in range(target.chosen_idx + 1, n_alts):
            p_alt, plan_alt, c2_alt = target.alternatives[alt_idx]

            # Restore state to "before target decision".
            fleet_restored = _clone_fleet(target.fleet_before)
            unrouted_restored = set(target.unrouted_before)

            # Apply this alternative in place of the original choice.
            apply_insertion(p_alt, plan_alt, fleet_restored)
            unrouted_restored.discard(p_alt)

            new_target = _Decision(
                iteration=target.iteration,
                fleet_before=target.fleet_before,
                unrouted_before=target.unrouted_before,
                alternatives=target.alternatives,
                chosen_idx=alt_idx,
            )
            new_decisions = list(decisions[:-depth]) + [new_target]

            log.info(
                "    try alt %d/%d at iter %d: %s  case=%s  c1=%.3f  c2=%.3f",
                alt_idx + 1, n_alts, target.iteration,
                p_alt, plan_alt.case, plan_alt.c1, c2_alt,
            )

            # Greedy-complete (no nested backtracking).
            status = _greedy_insert_loop(
                inst, cfg, fleet_restored, unrouted_restored,
                new_decisions, take_snapshots=False,
            )

            if status == "feasible":
                log.info(
                    "    backtrack SUCCESS at depth %d, alternative %d/%d (%s)",
                    depth, alt_idx + 1, n_alts, p_alt,
                )
                return fleet_restored, new_decisions, depth, alt_idx

        log.info(
            "  backtrack depth %d: all %d alternative(s) at iter %d failed",
            depth, n_remaining, target.iteration,
        )

    return None


def construct(inst, cfg: dict) -> tuple:
    """Run Solomon I1 construction. Returns (fleet, status, summary).

    status is "feasible" if every part is routed, or "infeasible_unrouted"
    if at least one part has no feasible insertion at termination.

    `summary["runtime_s"]` records wall-clock time spent in the
    construction (in seconds), measured from the first iteration through
    the final fleet evaluation. ``summary["backtrack"]`` records whether
    bounded-depth backtracking was attempted, and at which depth /
    alternative-index it succeeded (or that it failed).
    """
    import time as _time
    _t0 = _time.perf_counter()
    fleet = FleetState.empty(inst)
    unrouted = set(inst.P)
    decisions: list = []

    max_backtrack = int(cfg.get("solomon_backtrack_max_depth", 5))

    log.info(
        "Solomon construction: |P|=%d  |KR|=%d  wait_limit=%.2f  "
        "backtrack_max_depth=%d",
        len(unrouted), len(fleet.routes),
        cfg["heuristic_wait_limit"], max_backtrack,
    )

    # ---- Greedy pass (with snapshots if backtracking is enabled) ----
    status = _greedy_insert_loop(
        inst, cfg, fleet, unrouted, decisions,
        take_snapshots=(max_backtrack > 0),
    )

    backtrack_info = {
        "attempted": False,
        "succeeded": False,
        "succeeded_at_depth": None,
        "succeeded_at_alt_idx": None,
        "max_depth": max_backtrack,
    }

    # ---- Bounded-depth backtracking on infeasibility ----
    if status == "infeasible_unrouted" and max_backtrack > 0:
        routed = set()
        for r in fleet.routes.values():
            routed |= r.parts
        log.info(
            "Solomon greedy reached infeasibility (unrouted=%s); "
            "attempting bounded-depth backtrack (max_depth=%d)",
            sorted(set(inst.P) - routed), max_backtrack,
        )
        backtrack_info["attempted"] = True
        result = _try_backtrack(inst, cfg, fleet, decisions, max_backtrack)
        if result is not None:
            fleet, decisions, succ_depth, succ_alt_idx = result
            status = "feasible"
            backtrack_info["succeeded"] = True
            backtrack_info["succeeded_at_depth"] = succ_depth
            backtrack_info["succeeded_at_alt_idx"] = succ_alt_idx

    iteration = len(decisions)

    # ---- Bail out on still-infeasible ----
    if status == "infeasible_unrouted":
        routed = set()
        for r in fleet.routes.values():
            routed |= r.parts
        unrouted_final = sorted(set(inst.P) - routed)
        log.error(
            "No feasible insertion remains after backtracking; unrouted=%s",
            unrouted_final,
        )
        summary = {
            "status": "infeasible_unrouted",
            "unrouted": unrouted_final,
            "iterations": iteration,
            "route_duration": float("inf"),
            "total_wait": float("inf"),
            "runtime_s": _time.perf_counter() - _t0,
            "backtrack": backtrack_info,
        }
        return fleet, "infeasible_unrouted", summary

    # ---- Construction succeeded — phase snapshot ----
    construct_dur = fleet.total_route_duration()
    construct_wait = fleet.total_wait()
    construct_runtime = _time.perf_counter() - _t0
    log.info(
        "Solomon CONSTRUCT DONE: route_duration=%.3f  total_wait=%.3f  iters=%d  runtime=%.4fs",
        construct_dur, construct_wait, iteration, construct_runtime,
    )
    phases = [{
        "name": "construct",
        "f1": construct_dur,
        "f2": construct_wait,
        "runtime_s": construct_runtime,
        "iterations": iteration,
        "backtrack": dict(backtrack_info),
    }]

    # ---- Optional 2-opt local-search improvement phase ----
    if cfg.get("apply_2opt", False):
        _t_opt0 = _time.perf_counter()
        swaps_total = two_opt_fleet(fleet, cfg)
        two_opt_runtime = _time.perf_counter() - _t_opt0
        after_2opt_dur = fleet.total_route_duration()
        after_2opt_wait = fleet.total_wait()
        log.info(
            "Solomon 2-OPT DONE: swaps=%d  route_duration=%.3f  total_wait=%.3f  runtime=%.4fs",
            swaps_total, after_2opt_dur, after_2opt_wait, two_opt_runtime,
        )
        phases.append({
            "name": "2opt",
            "f1": after_2opt_dur,
            "f2": after_2opt_wait,
            "runtime_s": two_opt_runtime,
            "swaps": swaps_total,
        })

    # ---- Optional Or-opt-on-pairs improvement phase (final) ----
    if cfg.get("apply_or_opt", False):
        _t_or0 = _time.perf_counter()
        or_moves_total = or_opt_fleet(fleet, cfg)
        or_opt_runtime = _time.perf_counter() - _t_or0
        after_or_dur = fleet.total_route_duration()
        after_or_wait = fleet.total_wait()
        log.info(
            "Solomon OR-OPT DONE: moves=%d  route_duration=%.3f  total_wait=%.3f  runtime=%.4fs",
            or_moves_total, after_or_dur, after_or_wait, or_opt_runtime,
        )
        phases.append({
            "name": "or_opt",
            "f1": after_or_dur,
            "f2": after_or_wait,
            "runtime_s": or_opt_runtime,
            "moves": or_moves_total,
        })

    final_dur = fleet.total_route_duration()
    final_wait = fleet.total_wait()
    runtime_s = _time.perf_counter() - _t0
    log.info("Solomon DONE: route_duration=%.3f  total_wait=%.3f  runtime=%.4fs",
             final_dur, final_wait, runtime_s)
    summary = {
        "status": "feasible",
        "iterations": iteration,
        "route_duration": final_dur,
        "total_wait": final_wait,
        "runtime_s": runtime_s,
        "phases": phases,
        "backtrack": backtrack_info,
    }
    return fleet, "feasible", summary


# =============================================================================
# Multi-start construction (alpha-bias fallback)
# =============================================================================
# The single greedy construction in construct() is sensitive to the
# alpha-triple it's given.  On instances with tight shared-node patterns
# the primary bias for the targeted objective can fail even with deep
# backtracking, while a different bias produces a feasible (sometimes
# only mildly sub-optimal) route.  construct_with_multistart() tries the
# caller's primary bias first, then falls back to the other two stock
# biases until one succeeds.  This is the same idea NSGA-II's
# multi-decoding uses internally per chromosome.

_STOCK_BIASES = [
    # --- core three (used for both the primary attempt and basic fallback) ---
    ("route_duration", {"alpha_1": 0.0001, "alpha_2": 1.0,  "alpha_3": 0.01}),
    ("wait_time",      {"alpha_1": 0.0001, "alpha_2": 0.01, "alpha_3": 1.0}),
    ("balanced",       {"alpha_1": 0.0001, "alpha_2": 0.5,  "alpha_3": 0.5}),
    # --- extended diverse biases, tried only after the core three fail ---
    # Distance-heavy: scores insertions by the geometric closeness of the
    # inserted pair to existing route stops; tends to thread the needle on
    # tight ring routes where wait/duration biases create big skips.
    ("distance",       {"alpha_1": 1.0,    "alpha_2": 0.0001, "alpha_3": 0.0001}),
    # Distance + wait: secondary preference on low wait once neighbouring
    # arcs are short. Useful when a wait-dominant chain blocks a few late
    # products.
    ("distance_wait",  {"alpha_1": 0.5,    "alpha_2": 0.0001, "alpha_3": 0.5}),
    # Distance + duration: keeps geometric proximity but still pushes
    # towards short total fleet duration — good for tight single-route
    # instances where pure distance ignores total span.
    ("distance_dur",   {"alpha_1": 0.5,    "alpha_2": 0.5,    "alpha_3": 0.0001}),
]
_CORE_STOCK_INDICES = (0, 1, 2)   # indices of the 3 "core" biases above


def _bias_index_of(cfg: dict) -> int:
    """Return the index in _STOCK_BIASES whose alpha-triple matches cfg."""
    a2 = float(cfg.get("alpha_2", 1.0))
    a3 = float(cfg.get("alpha_3", 0.01))
    if abs(a3 - 1.0) < 1e-6 and abs(a2 - 0.01) < 1e-6:
        return 1   # wait_time bias
    if abs(a2 - 0.5) < 1e-6 and abs(a3 - 0.5) < 1e-6:
        return 2   # balanced bias
    return 0       # default: route-duration bias


def _sample_random_bias(rng) -> tuple:
    """Draw a random (name, alpha-triple) for a randomized restart.

    Samples each weight log-uniformly in ``[10**-4, 1]`` and normalises so
    they roughly sum to ~1, then names the bias for the dominant component.
    The very wide log-uniform range gives true diversity across restarts.
    """
    import math
    raw = tuple(10.0 ** rng.uniform(-4, 0) for _ in range(3))
    s = sum(raw) or 1.0
    a1, a2, a3 = (raw[0] / s, raw[1] / s, raw[2] / s)
    # Avoid pathological zeros that would degenerate the scoring sign.
    a1 = max(a1, 1e-6)
    a2 = max(a2, 1e-6)
    a3 = max(a3, 1e-6)
    dom = max(((a1, "dist"), (a2, "dur"), (a3, "wait")), key=lambda t: t[0])[1]
    name = f"random_{dom}"
    return name, {"alpha_1": a1, "alpha_2": a2, "alpha_3": a3}


def construct_with_multistart(inst, cfg: dict) -> tuple:
    """Drop-in replacement for construct() that adds bias fallback.

    Tries the caller's primary alpha-bias first (the one already encoded in
    cfg's alpha_1/2/3).  Each attempt uses ``solomon_backtrack_max_depth``
    from cfg, so it already includes bounded-depth backtracking before
    declaring infeasibility.  Fallback ladder (each rung adds robustness on
    tight pickup-delivery instances at minimal cost on easy ones):

      1. *Primary* bias — the one cfg dictates (route_duration or
         wait_time) with full bounded-depth backtracking.
      2. *Balanced* core bias — empirically the most likely to resolve
         shared-node deadlocks that a single-objective bias can't escape.
      3. The *other* single-objective core bias.
      4. *Extended* diverse biases (distance, distance+wait, distance+dur)
         — only tried if the three core biases all fail.  Distance-heavy
         starts often thread tight ring routes that wait/duration starts
         skip past.
      5. *Random* restarts — up to ``solomon_random_restarts`` log-uniform
         alpha triples drawn from a seeded RNG.  Tried only if every stock
         bias still fails.  ``solomon_random_seed`` (default 42) makes the
         sequence reproducible.

    Returns the first feasible (fleet, status, summary).  When every
    attempt fails, returns the summary from the primary attempt with
    ``summary["bias_used"]`` set to ``None``.

    ``summary["bias_used"]`` is set on every return so the xlsx writer can
    surface which bias produced the reported result.
    """
    primary_idx = _bias_index_of(cfg)
    BALANCED_IDX = 2   # core ordering: route_duration=0, wait_time=1, balanced=2
    if primary_idx == BALANCED_IDX:
        # Caller's primary already is balanced — fall back to the two
        # single-objective biases in canonical order.
        core_order = [BALANCED_IDX, 0, 1]
    else:
        # Primary is route_duration (0) or wait_time (1).  Try primary,
        # then balanced, then the other single-objective bias.
        other_idx = 1 - primary_idx
        core_order = [primary_idx, BALANCED_IDX, other_idx]

    # Extended biases come after the core three.  Indices 3, 4, 5.
    extended_order = [i for i in range(len(_STOCK_BIASES))
                      if i not in _CORE_STOCK_INDICES]
    order = core_order + extended_order

    first_fleet = None
    first_summary = None

    def _record_first(fleet, summary):
        nonlocal first_fleet, first_summary
        if first_summary is None:
            first_fleet, first_summary = fleet, summary

    # ---- Stock-bias ladder (core 3 + extended diverse triples) ----
    for trial_idx in order:
        name, alphas = _STOCK_BIASES[trial_idx]
        trial_cfg = dict(cfg)
        trial_cfg.update(alphas)
        fleet, status, summary = construct(inst, trial_cfg)
        summary["bias_used"] = name
        if status == "feasible":
            label = "primary" if trial_idx == primary_idx else (
                "core fallback" if trial_idx in _CORE_STOCK_INDICES
                else "extended fallback"
            )
            log.info("Solomon multi-start: success under bias=%s (%s)",
                     name, label)
            return fleet, status, summary
        log.info("Solomon multi-start: bias=%s infeasible; trying next bias",
                 name)
        _record_first(fleet, summary)

    # ---- Randomized alpha-bias restarts ----
    n_random = int(cfg.get("solomon_random_restarts", 0))
    if n_random > 0:
        import random
        seed = cfg.get("solomon_random_seed", 42)
        rng = random.Random(seed)
        log.info("Solomon multi-start: stock biases exhausted; "
                 "trying %d randomized restarts (seed=%s)", n_random, seed)
        for k in range(n_random):
            name, alphas = _sample_random_bias(rng)
            trial_cfg = dict(cfg)
            trial_cfg.update(alphas)
            log.info(
                "Solomon multi-start: random restart %d/%d  bias=%s  "
                "(a1=%.4f, a2=%.4f, a3=%.4f)",
                k + 1, n_random, name,
                alphas["alpha_1"], alphas["alpha_2"], alphas["alpha_3"],
            )
            fleet, status, summary = construct(inst, trial_cfg)
            summary["bias_used"] = f"{name}_seed{seed}_try{k+1}"
            if status == "feasible":
                log.info("Solomon multi-start: random restart %d/%d "
                         "produced a feasible construction (bias=%s)",
                         k + 1, n_random, name)
                return fleet, status, summary
            _record_first(fleet, summary)

    # ---- Randomized-ORDER construction (deepest LNS-style fallback) ----
    # When alpha-bias diversity fails, the structural deadlock is in the
    # *greedy c2 ordering*, not in the (c1, c2) weights themselves. Each
    # shuffled-order pass picks a uniformly random insertion order over
    # |P|, and at each step takes each part's best feasible insertion --
    # no global c2 ranking, no backtracking. Many shuffles defeat tight
    # ring instances where deterministic greedy creates an irrecoverable
    # early commitment. This is the same "ruin-and-recreate" intuition
    # as LNS but starting from an empty fleet for each random restart.
    n_shuf = int(cfg.get("solomon_shuffle_restarts", 0))
    if n_shuf > 0:
        import random
        seed = cfg.get("solomon_random_seed", 42)
        # Offset the seed so shuffle restarts don't reuse the alpha-bias
        # restart RNG stream.
        shuf_rng = random.Random((seed if seed is not None else 0) + 10_000)
        # Pick the alpha-bias that produced the *smallest* unrouted set
        # across all earlier attempts so far -- best heuristic guess at
        # which weighting still drives toward feasibility.
        best_bias_name, best_bias = _STOCK_BIASES[primary_idx]
        log.info("Solomon multi-start: alpha-bias restarts exhausted; "
                 "trying %d shuffled-order restarts under bias=%s "
                 "(seed=%s)", n_shuf, best_bias_name, seed)
        for s in range(n_shuf):
            trial_cfg = dict(cfg)
            trial_cfg.update(best_bias)
            fleet, status, summary = _construct_shuffled(
                inst, trial_cfg, shuf_rng,
            )
            tag = f"shuffle_seed{seed}_try{s+1}"
            summary["bias_used"] = tag
            if status == "feasible":
                log.info(
                    "Solomon multi-start: shuffled-order restart %d/%d "
                    "produced a feasible construction (bias=%s)",
                    s + 1, n_shuf, best_bias_name,
                )
                return fleet, status, summary
            if (s + 1) % 10 == 0 or s == 0:
                log.info("  shuffle %d/%d still failing  (unrouted=%s)",
                         s + 1, n_shuf, summary.get("unrouted"))
            _record_first(fleet, summary)

    first_summary["bias_used"] = None
    log.error("Solomon multi-start: all %d stock biases + %d random "
              "alpha-restart(s) + %d shuffled-order restart(s) "
              "exhausted without a feasible construction",
              len(_STOCK_BIASES), n_random, n_shuf)
    return first_fleet, "infeasible_unrouted", first_summary


def _construct_shuffled(inst, cfg: dict, rng) -> tuple:
    """One randomized-order Solomon pass.

    Inserts the |P| parts in a uniformly random order using best_insert_anywhere
    for each. No backtracking, no c2 ranking. Returns (fleet, status, summary)
    in the same shape as :func:`construct`. The summary's ``backtrack``
    sub-dict is filled with a "shuffle" marker so the xlsx writer can
    surface that this fallback was used.
    """
    _t0 = _time.perf_counter()
    fleet = FleetState.empty(inst)
    parts = list(inst.P)
    rng.shuffle(parts)
    lam = cfg["lambda_c2"]
    inserted = 0
    for p in parts:
        baseline_dur = fleet.total_route_duration()
        baseline_wait = fleet.total_wait()
        plan = best_insert_anywhere(p, fleet, cfg, baseline_dur, baseline_wait)
        if not math.isfinite(plan.c1):
            # This shuffle failed at part p; nothing more to do.
            routed_set = set()
            for r in fleet.routes.values():
                routed_set |= r.parts
            unrouted_final = sorted(set(inst.P) - routed_set)
            summary = {
                "status": "infeasible_unrouted",
                "unrouted": unrouted_final,
                "iterations": inserted,
                "route_duration": float("inf"),
                "total_wait": float("inf"),
                "runtime_s": _time.perf_counter() - _t0,
                "backtrack": {
                    "attempted": False,
                    "succeeded": False,
                    "max_depth": 0,
                    "shuffle_used": True,
                },
            }
            return fleet, "infeasible_unrouted", summary
        apply_insertion(p, plan, fleet)
        inserted += 1

    # All parts routed under this shuffle. Run optional local-search
    # improvements so the returned solution matches construct()'s shape.
    construct_dur = fleet.total_route_duration()
    construct_wait = fleet.total_wait()
    construct_rt = _time.perf_counter() - _t0
    phases = [{
        "name": "construct",
        "f1": construct_dur,
        "f2": construct_wait,
        "runtime_s": construct_rt,
        "iterations": inserted,
        "backtrack": {
            "attempted": False, "succeeded": False,
            "max_depth": 0, "shuffle_used": True,
        },
    }]
    if cfg.get("apply_2opt", False):
        _t1 = _time.perf_counter()
        swaps = two_opt_fleet(fleet, cfg)
        phases.append({
            "name": "2opt",
            "f1": fleet.total_route_duration(),
            "f2": fleet.total_wait(),
            "runtime_s": _time.perf_counter() - _t1,
            "swaps": swaps,
        })
    if cfg.get("apply_or_opt", False):
        _t2 = _time.perf_counter()
        moves = or_opt_fleet(fleet, cfg)
        phases.append({
            "name": "or_opt",
            "f1": fleet.total_route_duration(),
            "f2": fleet.total_wait(),
            "runtime_s": _time.perf_counter() - _t2,
            "moves": moves,
        })
    summary = {
        "status": "feasible",
        "iterations": inserted,
        "route_duration": fleet.total_route_duration(),
        "total_wait": fleet.total_wait(),
        "runtime_s": _time.perf_counter() - _t0,
        "phases": phases,
        "backtrack": {
            "attempted": False, "succeeded": False,
            "max_depth": 0, "shuffle_used": True,
        },
    }
    return fleet, "feasible", summary


# =============================================================================
# 2-opt local search
# =============================================================================
# A 2-opt move on a single route replaces two non-adjacent arcs by their
# "uncrossed" counterparts, equivalently reversing the segment between
# them. For a route [h, ..., a, b, ..., c, d, ..., h] with arcs (a,b) and
# (c,d), the move yields [h, ..., a, c, ..., b, d, ..., h]. In PD-VRP the
# reversal can place a delivery before its pickup, so every candidate move
# is rebuilt as a RouteState and run through simulate_route to check
# feasibility (precedence, capacity, shift window). The move is accepted
# only if it is feasible AND strictly improves the heuristic_objective.

def _route_primary_value(route: RouteState, inst, T_start: float,
                          objective: str) -> float:
    """Score a candidate route by the primary objective (legacy helper)."""
    sim = simulate_route(route, inst, T_start=T_start)
    if not sim.get("feasible", False):
        return float("inf")
    if objective == "wait_time":
        return sum(sim["waits"].values())
    # default: route_duration  =  depot return time on this single route
    return sim["depot_arrival"]


def _route_lex_score(route: RouteState, inst, T_start: float, cfg: dict
                      ) -> tuple:
    """Lex-aware score used by 2-opt and Or-opt comparators.

    Returns (score, sim) where score is the alpha-weighted combination

        alpha_1 * total_distance
      + alpha_2 * route_duration   (depot_arrival on this route)
      + alpha_3 * total_wait

    using the SAME alpha-triple as the c1 cost in construction. With the
    strict-lex defaults set in run_model.TEST_ALPHAS, this gives:

      heuristic_objective = route_duration  ->  (0.0001, 1,    0.01)
      heuristic_objective = wait_time       ->  (0.0001, 0.01, 1)

    so the primary objective dominates, the secondary objective is a
    tiebreaker, and geometric distance is a third-level tiebreaker — the
    same hierarchy construction uses.

    Returns (inf, infeasible_sim) when the route is infeasible.
    """
    sim = simulate_route(route, inst, T_start=T_start)
    if not sim.get("feasible", False):
        return float("inf"), sim
    a1 = cfg.get("alpha_1", 0.0)
    a2 = cfg.get("alpha_2", 0.0)
    a3 = cfg.get("alpha_3", 0.0)
    # Distance = sum of arc costs along route.nodes (depot included).
    nodes = route.nodes
    dist = 0.0
    for i in range(len(nodes) - 1):
        dist += inst.c.get((nodes[i], nodes[i + 1]), 0.0)
    score = (a1 * dist
              + a2 * sim["depot_arrival"]
              + a3 * sum(sim["waits"].values()))
    return score, sim


def two_opt_route(route: RouteState, inst, T_start: float, cfg: dict
                   ) -> tuple:
    """First-improvement 2-opt on one route. Returns (n_swaps, final_route).

    Move acceptance uses the lex-weighted score from _route_lex_score, so
    among moves that improve the primary objective the one that also
    improves (or doesn't worsen) the secondary objective wins. With the
    strict-lex defaults the heuristic will not trade f2 to improve f1 by
    a tiny amount; the secondary axis is a true tiebreaker. Wait-limit
    constraint is honoured.
    """
    wait_limit = cfg.get("heuristic_wait_limit", float("inf"))
    swaps = 0
    improved = True
    while improved:
        improved = False
        nodes = route.nodes
        n = len(nodes)
        if n < 5:
            break
        cur_score, cur_sim = _route_lex_score(route, inst, T_start, cfg)
        if not cur_sim.get("feasible", False):
            break
        cur_f1 = cur_sim["depot_arrival"]
        cur_f2 = sum(cur_sim["waits"].values())
        for i in range(n - 3):
            for j in range(i + 2, n - 1):
                new_nodes = (nodes[:i + 1]
                             + list(reversed(nodes[i + 1:j + 1]))
                             + nodes[j + 1:])
                trial = RouteState(k=route.k, r=route.r)
                trial.nodes = new_nodes
                trial.parts = set(route.parts)
                new_score, sim = _route_lex_score(trial, inst, T_start, cfg)
                if not sim.get("feasible", False):
                    continue
                new_wait_total = sum(sim["waits"].values())
                if new_wait_total > wait_limit + EPS:
                    continue
                if new_score + EPS < cur_score:
                    log.info(
                        "2-opt swap on (%s, R%d): reverse [%d..%d] "
                        "'%s' -> '%s',  f1: %.3f -> %.3f,  f2: %.3f -> %.3f",
                        route.k, route.r, i + 1, j,
                        " ".join(nodes), " ".join(new_nodes),
                        cur_f1, sim["depot_arrival"],
                        cur_f2, new_wait_total,
                    )
                    route.nodes = new_nodes
                    swaps += 1
                    improved = True
                    break
            if improved:
                break
    return swaps, route


def or_opt_route(route: RouteState, inst, T_start: float, cfg: dict
                  ) -> tuple:
    """First-improvement Or-opt on pairs for one route."""
    wait_limit = cfg.get("heuristic_wait_limit", float("inf"))
    moves = 0
    improved = True
    while improved:
        improved = False
        for p in sorted(route.parts):
            op = inst.o[p]
            dp = inst.d[p]
            o_pos = route.node_position(op)
            d_pos = route.node_position(dp)
            if o_pos < 0 or d_pos < 0:
                continue
            shared_o = any(
                inst.o[q] == op or inst.d[q] == op
                for q in route.parts if q != p
            )
            shared_d = any(
                inst.o[q] == dp or inst.d[q] == dp
                for q in route.parts if q != p
            )
            if shared_o or shared_d:
                continue

            hi, lo = (d_pos, o_pos) if d_pos > o_pos else (o_pos, d_pos)
            base_nodes = list(route.nodes)
            del base_nodes[hi]
            del base_nodes[lo]
            n = len(base_nodes)
            cur_score, cur_sim = _route_lex_score(route, inst, T_start, cfg)
            if not cur_sim.get("feasible", False):
                continue
            cur_f1 = cur_sim["depot_arrival"]
            cur_f2 = sum(cur_sim["waits"].values())

            best_score = cur_score
            best_pair = None
            best_f1 = cur_f1
            best_f2 = cur_f2

            for new_o_pos in range(1, n):
                interim = base_nodes[:new_o_pos] + [op] + base_nodes[new_o_pos:]
                for new_d_pos in range(new_o_pos + 1, n + 1):
                    trial_nodes = (interim[:new_d_pos] + [dp]
                                    + interim[new_d_pos:])
                    if (new_o_pos == o_pos and
                            new_d_pos == (d_pos if d_pos > o_pos else d_pos + 1)):
                        continue
                    trial = RouteState(k=route.k, r=route.r)
                    trial.nodes = trial_nodes
                    trial.parts = set(route.parts)
                    new_score, sim = _route_lex_score(trial, inst, T_start, cfg)
                    if not sim.get("feasible", False):
                        continue
                    new_wait_total = sum(sim["waits"].values())
                    if new_wait_total > wait_limit + EPS:
                        continue
                    if new_score + EPS < best_score:
                        best_score = new_score
                        best_pair = (new_o_pos, new_d_pos)
                        best_f1 = sim["depot_arrival"]
                        best_f2 = new_wait_total

            if best_pair is not None and best_score + EPS < cur_score:
                new_o_pos, new_d_pos = best_pair
                interim = base_nodes[:new_o_pos] + [op] + base_nodes[new_o_pos:]
                new_nodes = interim[:new_d_pos] + [dp] + interim[new_d_pos:]
                route.nodes = new_nodes
                moves += 1
                improved = True
                break
    return moves, route


def or_opt_fleet(fleet: FleetState, cfg: dict) -> int:
    inst = fleet.inst
    total_moves = 0
    for k in inst.K:
        T_current = 0.0
        for route in fleet.routes_of_vehicle(k):
            if route.is_empty():
                continue
            moves, _ = or_opt_route(route, inst, T_current, cfg)
            total_moves += moves
            sim = simulate_route(route, inst, T_start=T_current)
            if sim.get("feasible", False):
                T_current = sim["depot_arrival"]
            else:
                break
        fleet.invalidate(k)
    return total_moves


def two_opt_fleet(fleet: FleetState, cfg: dict) -> int:
    inst = fleet.inst
    total_swaps = 0
    for k in inst.K:
        T_current = 0.0
        for route in fleet.routes_of_vehicle(k):
            if route.is_empty():
                continue
            swaps, _ = two_opt_route(route, inst, T_current, cfg)
            total_swaps += swaps
            sim = simulate_route(route, inst, T_start=T_current)
            if sim.get("feasible", False):
                T_current = sim["depot_arrival"]
            else:
                break
        fleet.invalidate(k)
    return total_swaps


def fleet_to_solution(fleet: FleetState) -> Solution:
    inst = fleet.inst
    x_used = set(); f_assigned = {}
    ta = {}; td = {}; ts = {}; y = {}; u = {}; delta = {}
    w = {p: 0.0 for p in inst.P}
    z_used = {}
    for k in inst.K:
        sims = fleet.simulate_vehicle(k)
        for sim_idx, route in enumerate(fleet.routes_of_vehicle(k)):
            r = route.r
            sim = sims[sim_idx]
            used = (not route.is_empty()) and sim.get("feasible", False)
            z_used[(k, r)] = used
            if not sim.get("feasible", False):
                continue
            if not route.is_empty():
                for a, b in zip(route.nodes[:-1], route.nodes[1:]):
                    x_used.add((a, b, k, r))
            for p in route.parts:
                f_assigned[p] = (k, r)
                w[p] = sim["waits"][p]
            ta.update(sim["ta"]); td.update(sim["td"]); ts.update(sim["ts"]); y.update(sim["y"])
            for rank, node in enumerate(route.customer_nodes(), start=1):
                u[(node, k, r)] = rank
                pickup_area = sum(inst.q_p[p] for p in route.parts if inst.o[p] == node)
                deliv_area = sum(inst.q_p[p] for p in route.parts if inst.d[p] == node)
                delta[(node, k, r)] = pickup_area - deliv_area
    for (k, r) in inst.KR_pairs:
        for j in inst.Nw:
            u.setdefault((j, k, r), 0)
            delta.setdefault((j, k, r), 0.0)
    return Solution(
        x_used=x_used, f_assigned=f_assigned,
        ta=ta, td=td, ts=ts, w=w, y=y, u=u, delta=delta,
        z_used=z_used,
        route_duration=fleet.total_route_duration(),
        total_wait=fleet.total_wait(),
    )
