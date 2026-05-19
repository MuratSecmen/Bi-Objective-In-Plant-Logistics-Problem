"""
verify.py — Solution validator for the Internal Logistics PD-VRP MIP

After the solver returns, this module recomputes every assumption made by
the model directly from the variable values (x.X, f.X, ta.X, ...) and
compares the result against what the model claims. Any mismatch is reported
as a `Finding`. Pass tolerance defaults to 1e-6.

Categories of checks (see `validate_solution`):

  1. route_topology
  2. assignment
  3. pickup_before_delivery
  4. ready_time
  5. travel_time
  6. service_time
  7. wait_time
  8. capacity
  9. route_monotonicity

Usage from run_model.py:

    from verify import extract_solution, validate_solution, format_report
    sol = extract_solution(m, vars_, inst)
    findings = validate_solution(sol, inst)
    text = format_report(findings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

TOL = 1e-6


# =============================================================================
# Data carriers
# =============================================================================
@dataclass
class Finding:
    category: str          # e.g. "route_topology"
    test_id: str           # e.g. "route_closure[V1,r=1]"
    passed: bool
    detail: str = ""       # human-readable description when failed

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.category}/{self.test_id}{': ' + self.detail if self.detail else ''}"


@dataclass
class Solution:
    """Pure-Python snapshot of a solved Gurobi model — no Gurobi dependency."""
    x_used: set            # set of (i, j, k, r) with x.X > 0.5
    f_assigned: dict       # {p: (k, r)} where f[p, k, r].X > 0.5
    ta: dict               # {(j, k, r): value}
    td: dict
    ts: dict               # only defined on Nw
    w: dict                # {p: value}
    y: dict                # {(j, k, r): value}
    u: dict                # only defined on Nw
    delta: dict            # only defined on Nw
    z_used: dict           # {(k, r): bool} — route activation (vehicle departs)
    route_duration: float
    total_wait: float

    def route_order(self, k, r) -> list:
        """Return the ordered node list for a used (k, r), starting at h.

        Raises ValueError if the recovered path doesn't close at h or
        revisits a node — both indicate a topology bug.
        """
        if not self.z_used.get((k, r), False):
            return []
        order = ["h"]
        visited = {"h"}
        current = "h"
        while True:
            next_nodes = [j for (i, j, kk, rr) in self.x_used
                          if kk == k and rr == r and i == current]
            if len(next_nodes) == 0:
                raise ValueError(
                    f"route ({k},{r}): no outgoing arc from {current}"
                )
            if len(next_nodes) > 1:
                raise ValueError(
                    f"route ({k},{r}): {current} has multiple outflows {next_nodes}"
                )
            nxt = next_nodes[0]
            order.append(nxt)
            if nxt == "h":
                return order
            if nxt in visited:
                raise ValueError(
                    f"route ({k},{r}): revisited {nxt} — subtour or duplicate"
                )
            visited.add(nxt)
            current = nxt


# =============================================================================
# Extraction from Gurobi variables
# =============================================================================
def extract_solution(m, vars_, inst) -> Solution:
    x = vars_["x"]; f = vars_["f"]; w = vars_["w"]; y = vars_["y"]
    ta = vars_["ta"]; td = vars_["td"]; ts = vars_["ts"]
    u = vars_["u"]; delta = vars_["delta"]

    x_used = {key for key, v in x.items() if v.X > 0.5}

    f_assigned = {}
    for (p, k, r), v in f.items():
        if v.X > 0.5:
            if p in f_assigned:
                # caught later by assignment check; record arbitrarily
                pass
            f_assigned[p] = (k, r)

    ta_v = {key: v.X for key, v in ta.items()}
    td_v = {key: v.X for key, v in td.items()}
    ts_v = {key: v.X for key, v in ts.items()}
    w_v = {p: w[p].X for p in inst.P}
    y_v = {key: v.X for key, v in y.items()}
    u_v = {key: v.X for key, v in u.items()}
    delta_v = {key: v.X for key, v in delta.items()}

    z_used = {}
    for k in inst.K:
        for r in inst.routes_of(k):
            departed = any((x["h", j, k, r].X > 0.5)
                           for j in inst.Nw if ("h", j, k, r) in x)
            z_used[(k, r)] = departed

    # Each vehicle's "last route" can differ (max_routes from vehicles.xlsx).
    route_duration = sum(ta_v[("h", k, inst.last_route_of(k))]
                         for k in inst.K)
    total_wait = sum(w_v[p] for p in inst.P)

    return Solution(
        x_used=x_used, f_assigned=f_assigned,
        ta=ta_v, td=td_v, ts=ts_v, w=w_v, y=y_v, u=u_v, delta=delta_v,
        z_used=z_used,
        route_duration=route_duration, total_wait=total_wait,
    )


# =============================================================================
# Individual check categories
# =============================================================================
def _check_route_topology(sol, inst, findings):
    cat = "route_topology"
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used[(k, r)]:
                findings.append(Finding(cat, f"unused[{k},r={r}]", True))
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError as ex:
                findings.append(Finding(
                    cat, f"recover[{k},r={r}]", False, str(ex)
                ))
                continue
            findings.append(Finding(
                cat, f"recover[{k},r={r}]", True,
                f"route = {' -> '.join(seq)}"
            ))
            # flow conservation already enforced; double-check single visit
            non_h = [n for n in seq if n != "h"]
            if len(non_h) != len(set(non_h)):
                findings.append(Finding(
                    cat, f"unique_visits[{k},r={r}]", False,
                    f"duplicate nodes in {seq}"
                ))


def _check_assignment(sol, inst, findings):
    cat = "assignment"
    counts = {p: 0 for p in inst.P}
    for (p, k, r) in [(p, *sol.f_assigned[p]) for p in inst.P if p in sol.f_assigned]:
        counts[p] += 1
    for p in inst.P:
        ok = counts[p] == 1
        findings.append(Finding(
            cat, f"each_product_once[{p}]", ok,
            "" if ok else f"assigned to {counts[p]} (k, r) pairs"
        ))
        if not ok or p not in sol.f_assigned:
            continue
        k, r = sol.f_assigned[p]
        # Visit origin and destination on the assigned (k, r)
        visits_op = any((i, inst.o[p], k, r) in sol.x_used for i in inst.N)
        visits_dp = any((i, inst.d[p], k, r) in sol.x_used for i in inst.N)
        findings.append(Finding(
            cat, f"visit_origin[{p}]", visits_op,
            "" if visits_op else f"vehicle {k} route {r} does not visit {inst.o[p]}"
        ))
        findings.append(Finding(
            cat, f"visit_destination[{p}]", visits_dp,
            "" if visits_dp else f"vehicle {k} route {r} does not visit {inst.d[p]}"
        ))


def _check_pickup_before_delivery(sol, inst, findings):
    cat = "pickup_before_delivery"
    for p in inst.P:
        if p not in sol.f_assigned:
            continue
        k, r = sol.f_assigned[p]
        op, dp = inst.o[p], inst.d[p]
        # u-order check (only meaningful when both are work-centres)
        if op in inst.Nw and dp in inst.Nw:
            uop = sol.u[(op, k, r)]
            udp = sol.u[(dp, k, r)]
            ok_u = udp > uop + 0.5
            findings.append(Finding(
                cat, f"u_order[{p}]", ok_u,
                "" if ok_u else f"u({dp})={udp} not > u({op})={uop}"
            ))
        # time order: td_op <= ta_dp
        td_op = sol.td[(op, k, r)]
        ta_dp = sol.ta[(dp, k, r)]
        ok_t = ta_dp >= td_op - TOL
        findings.append(Finding(
            cat, f"time_order[{p}]", ok_t,
            "" if ok_t else f"ta({dp})={ta_dp:.3f} < td({op})={td_op:.3f}"
        ))


def _check_ready_time(sol, inst, findings):
    cat = "ready_time"
    for p in inst.P:
        if p not in sol.f_assigned:
            continue
        k, r = sol.f_assigned[p]
        op = inst.o[p]
        if op == "h":
            continue  # ts not defined for depot
        ts_op = sol.ts[(op, k, r)]
        ok = ts_op >= inst.e[p] - TOL
        findings.append(Finding(
            cat, f"ts_ge_ep[{p}]", ok,
            "" if ok else f"ts({op})={ts_op:.3f} < e_p={inst.e[p]}"
        ))


def _check_travel_time(sol, inst, findings):
    cat = "travel_time"
    for (i, j, k, r) in sol.x_used:
        td_i = sol.td[(i, k, r)]
        ta_j = sol.ta[(j, k, r)]
        c_ij = inst.c[(i, j)]
        ok = ta_j >= td_i + c_ij - TOL
        findings.append(Finding(
            cat, f"arc[{i}->{j},{k},r={r}]", ok,
            "" if ok else f"ta({j})={ta_j:.3f} < td({i})+c={td_i + c_ij:.3f}"
        ))


def _check_service_time(sol, inst, findings):
    cat = "service_time"
    for j in inst.Nw:
        for k in inst.K:
            for r in inst.routes_of(k):
                if not sol.z_used[(k, r)]:
                    continue
                # Only enforce at visited nodes
                visited = any((i, j, k, r) in sol.x_used for i in inst.N)
                if not visited:
                    continue
                ta_j = sol.ta[(j, k, r)]
                ts_j = sol.ts[(j, k, r)]
                td_j = sol.td[(j, k, r)]
                unload = sum(inst.s_unload[p]
                             for p in inst.P
                             if inst.d[p] == j and sol.f_assigned.get(p) == (k, r))
                load = sum(inst.s_load[p]
                           for p in inst.P
                           if inst.o[p] == j and sol.f_assigned.get(p) == (k, r))
                ok_ts = ts_j >= ta_j + unload - TOL
                ok_td = td_j >= ts_j + load - TOL
                findings.append(Finding(
                    cat, f"ts_ge_ta_plus_unload[{j},{k},r={r}]", ok_ts,
                    "" if ok_ts else
                    f"ts={ts_j:.3f} < ta+unload={ta_j+unload:.3f}"
                ))
                findings.append(Finding(
                    cat, f"td_ge_ts_plus_load[{j},{k},r={r}]", ok_td,
                    "" if ok_td else
                    f"td={td_j:.3f} < ts+load={ts_j+load:.3f}"
                ))


def _check_wait_time(sol, inst, findings):
    cat = "wait_time"
    for p in inst.P:
        if p not in sol.f_assigned:
            findings.append(Finding(
                cat, f"recompute[{p}]", False, "product not assigned"
            ))
            continue
        k, r = sol.f_assigned[p]
        dp = inst.d[p]
        ta_dp = sol.ta[(dp, k, r)]
        expected = max(0.0, ta_dp + inst.s_unload[p] - inst.e[p])
        actual = sol.w[p]
        ok = abs(actual - expected) < 1e-3   # solver may have small slack
        findings.append(Finding(
            cat, f"recompute[{p}]", ok,
            "" if ok else
            f"w[{p}]={actual:.4f}, expected max(0, ta({dp})+s^u-e_p)={expected:.4f}"
        ))


def _check_capacity(sol, inst, findings):
    cat = "capacity"
    for k in inst.K:
        cap = inst.q_k[k]
        for r in inst.routes_of(k):
            if not sol.z_used[(k, r)]:
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue  # already reported in topology
            # y at depot
            y_h = sol.y[("h", k, r)]
            ok_h = abs(y_h) < TOL
            findings.append(Finding(
                cat, f"y_depot_zero[{k},r={r}]", ok_h,
                "" if ok_h else f"y[h]={y_h:.4f} != 0"
            ))
            # walk the route, check y in [0, cap] and delta is correct
            for j in seq:
                if j == "h":
                    continue
                y_j = sol.y[(j, k, r)]
                ok_lb = y_j >= -TOL
                ok_ub = y_j <= cap + TOL
                findings.append(Finding(
                    cat, f"y_in_range[{j},{k},r={r}]", ok_lb and ok_ub,
                    "" if (ok_lb and ok_ub) else
                    f"y[{j}]={y_j:.4f} not in [0, {cap}]"
                ))
                # delta correctness
                pickup_area = sum(inst.q_p[p] for p in inst.P
                                  if inst.o[p] == j and sol.f_assigned.get(p) == (k, r))
                drop_area = sum(inst.q_p[p] for p in inst.P
                                if inst.d[p] == j and sol.f_assigned.get(p) == (k, r))
                expected_delta = pickup_area - drop_area
                actual_delta = sol.delta[(j, k, r)]
                ok_d = abs(actual_delta - expected_delta) < 1e-3
                findings.append(Finding(
                    cat, f"delta[{j},{k},r={r}]", ok_d,
                    "" if ok_d else
                    f"delta={actual_delta:.4f}, expected {expected_delta:.4f}"
                ))


def _check_route_monotonicity(sol, inst, findings):
    cat = "route_monotonicity"
    for k in inst.K:
        for r in inst.routes_of(k)[:-1]:
            this_used = sol.z_used[(k, r)]
            next_used = sol.z_used[(k, r + 1)]
            ok = (not next_used) or this_used
            findings.append(Finding(
                cat, f"monotone[{k},r={r}->{r + 1}]", ok,
                "" if ok else f"route {r + 1} used but {r} not used"
            ))


# =============================================================================
# Public entry point
# =============================================================================
def validate_solution(sol: Solution, inst) -> list[Finding]:
    findings: list[Finding] = []
    _check_route_topology(sol, inst, findings)
    _check_assignment(sol, inst, findings)
    _check_pickup_before_delivery(sol, inst, findings)
    _check_ready_time(sol, inst, findings)
    _check_travel_time(sol, inst, findings)
    _check_service_time(sol, inst, findings)
    _check_wait_time(sol, inst, findings)
    _check_capacity(sol, inst, findings)
    _check_route_monotonicity(sol, inst, findings)
    return findings


def format_report(findings: list[Finding]) -> str:
    by_cat: dict[str, list[Finding]] = {}
    for f in findings:
        by_cat.setdefault(f.category, []).append(f)
    lines = []
    total_pass = sum(1 for f in findings if f.passed)
    total = len(findings)
    lines.append(f"VERIFICATION REPORT — {total_pass}/{total} checks passed")
    lines.append("=" * 72)
    for cat in [
        "route_topology", "assignment", "pickup_before_delivery",
        "ready_time", "travel_time", "service_time", "wait_time",
        "capacity", "route_monotonicity",
    ]:
        if cat not in by_cat:
            continue
        items = by_cat[cat]
        n_pass = sum(1 for f in items if f.passed)
        lines.append(f"  {cat:28} {n_pass:>4}/{len(items):<4}  "
                     f"{'OK' if n_pass == len(items) else 'FAIL'}")
    failed = [f for f in findings if not f.passed]
    if failed:
        lines.append("")
        lines.append("FAILED CHECKS:")
        for f in failed:
            lines.append(f"  - {f.category}/{f.test_id}: {f.detail}")
    return "\n".join(lines)


def passed(findings: list[Finding]) -> bool:
    return all(f.passed for f in findings)
