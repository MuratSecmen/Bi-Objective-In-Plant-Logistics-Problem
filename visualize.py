"""
visualize.py — Solution visualization for the Internal Logistics PD-VRP MIP

Three plots per run, all saved as PNG next to the result xlsx:

  - Gantt chart per vehicle      -> plot_gantt(sol, inst, out_path)
  - Route node-sequence diagram  -> plot_route_diagram(sol, inst, out_path)
  - Waiting-time bar chart       -> plot_wait_bars(sol, inst, out_path)

Matplotlib is used throughout. No external GIS data is required: the route
diagram uses a circular layout with the depot at the centre.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / server / Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Distinct, colour-blind-friendly palette
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _vehicle_colour(vehicle_index: int) -> str:
    return PALETTE[vehicle_index % len(PALETTE)]


def _route_colour(route_index: int) -> str:
    # within a vehicle, vary lightness to distinguish routes
    return PALETTE[route_index % len(PALETTE)]


def _hhmm(minutes: float, offset: int) -> str:
    minutes = minutes + offset
    h = int(minutes // 60)
    m = int(round(minutes - 60 * h))
    return f"{h:02d}:{m:02d}"


def carrying_on_arcs(sol, inst, k, r) -> list:
    """Return [(i, j, frozenset_of_product_ids_aboard), ...] for each arc of
    a used route (k, r). The aboard set captures what the vehicle holds
    WHILE TRAVERSING the arc (after applying drops + pickups at the origin
    node of the arc, before reaching the destination node).

    Returns empty list if (k, r) is not used or the path cannot be recovered.
    """
    if not sol.z_used.get((k, r), False):
        return []
    try:
        seq = sol.route_order(k, r)
    except ValueError:
        return []
    products_on_kr = [p for p, (kk, rr) in sol.f_assigned.items()
                      if kk == k and rr == r]
    aboard = set()
    out = []
    for idx in range(len(seq) - 1):
        i, j = seq[idx], seq[idx + 1]
        # Apply node-i events before the vehicle leaves: drop then pickup.
        # (At the depot node "h" both lists are empty; this is benign.)
        aboard -= {p for p in products_on_kr if inst.d[p] == i}
        aboard |= {p for p in products_on_kr if inst.o[p] == i}
        out.append((i, j, frozenset(aboard)))
    return out


# =============================================================================
# Gantt chart
# =============================================================================
def plot_gantt(sol, inst, out_path: Path) -> None:
    """Horizontal Gantt: one band per vehicle, segments for each used route.

    Each route is rendered as three layered components on the same lane:
      - travel arcs (vehicle colour, hatched diagonal)
      - service intervals at visited nodes (solid vehicle colour)
      - waiting intervals at origins before ready time (light grey)
    """
    fig, ax = plt.subplots(figsize=(12, max(3, 1.4 * len(inst.K) + 1)))

    bar_height = 0.8
    offset = inst.shift_offset

    for vi, k in enumerate(inst.K):
        colour = _vehicle_colour(vi)
        y_centre = len(inst.K) - vi - 1
        ax.text(-15, y_centre, k, va="center", ha="right",
                fontsize=11, fontweight="bold")

        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue

            # Walk the route and render each leg
            for i_node, node in enumerate(seq):
                if node == "h":
                    if i_node == len(seq) - 1:
                        break
                    next_node = seq[i_node + 1]
                    td_h = sol.td[(node, k, r)]
                    ta_next = sol.ta[(next_node, k, r)]
                    ax.barh(y_centre, ta_next - td_h, left=td_h,
                            height=bar_height, color=colour,
                            alpha=0.35, edgecolor="black", linewidth=0.5,
                            hatch="///")
                    continue

                # Service block at this node
                ta = sol.ta[(node, k, r)]
                ts = sol.ts[(node, k, r)]
                td = sol.td[(node, k, r)]

                # Optional wait-before-service block (ta -> ts is unload time;
                # actual idle waiting at origin is ts - ta for unloading and
                # ts >= e_p enforcement may push it later)
                if ts - ta > 0.001:
                    ax.barh(y_centre, ts - ta, left=ta,
                            height=bar_height, color="#dddddd",
                            edgecolor="grey", linewidth=0.3)
                if td - ts > 0.001:
                    ax.barh(y_centre, td - ts, left=ts,
                            height=bar_height, color=colour,
                            alpha=0.95, edgecolor="black", linewidth=0.5)

                # Travel to next node if not last
                if i_node < len(seq) - 1:
                    next_node = seq[i_node + 1]
                    ta_next = sol.ta[(next_node, k, r)]
                    ax.barh(y_centre, ta_next - td, left=td,
                            height=bar_height, color=colour,
                            alpha=0.35, edgecolor="black", linewidth=0.5,
                            hatch="///")

                # Node label above the bar
                mid_x = (ta + td) / 2
                ax.text(mid_x, y_centre + bar_height / 2 + 0.05, node,
                        ha="center", va="bottom", fontsize=8)

            # Mark route end with a small triangle at ta[h, k, r]
            ta_back = sol.ta[("h", k, r)]
            ax.plot(ta_back, y_centre, marker="v", color=colour,
                    markersize=10, markeredgecolor="black")

    # Axes
    ax.set_yticks([len(inst.K) - vi - 1 for vi in range(len(inst.K))])
    ax.set_yticklabels(["" for _ in inst.K])
    ax.set_ylim(-0.7, len(inst.K) - 0.3)
    ax.set_xlim(left=0)

    # Bottom: shift-relative minutes
    ax.set_xlabel("Minutes since shift start")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Top: clock time
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    xticks = ax.get_xticks()
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels([_hhmm(t, offset) for t in xticks])
    ax_top.set_xlabel("Clock time")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="grey", alpha=0.3, edgecolor="black",
                       hatch="///", label="travel"),
        mpatches.Patch(facecolor="grey", alpha=0.9, label="service (load+unload)"),
        mpatches.Patch(facecolor="#dddddd", edgecolor="grey",
                       label="wait at node"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    # |R| label shows the per-vehicle route counts since they can differ.
    r_summary = ",".join(str(inst.max_routes[k]) for k in inst.K)
    ax.set_title(
        f"Vehicle Gantt — case={inst.config['product_set_id']}, "
        f"|P|={len(inst.P)}, |K|={len(inst.K)}, R/veh=[{r_summary}]",
        fontsize=12, pad=20,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Route node-sequence diagrams
# =============================================================================
def plot_route_diagram(sol, inst, out_path: Path) -> None:
    """One subplot per USED (k, r): circular layout with depot at centre.

    Without geographic coordinates, we place work-centres equally spaced on
    a circle and the depot at the origin. Arrows are drawn for each arc in
    the route; their colour matches the vehicle.
    """
    used = [(k, r) for (k, r) in inst.KR_pairs if sol.z_used.get((k, r), False)]
    if not used:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No routes used", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    n_plots = len(used)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    # Pre-compute polar positions of work-centres (same layout in every subplot)
    Nw_sorted = sorted(inst.Nw, key=lambda n: int(n[1:]) if n[1:].isdigit() else 0)
    n_w = len(Nw_sorted)

    # Layout scale grows with the number of work-stations so the chord
    # between adjacent nodes stays comfortable (target ~0.55 data units).
    # Floor at 1.0 keeps small instances looking like before; ceiling at
    # 3.0 prevents absurdly large figures for very dense node sets.
    if n_w >= 2:
        target_chord = 0.55
        scale = target_chord / (2.0 * math.sin(math.pi / n_w))
    else:
        scale = 1.0
    scale = max(1.0, min(3.0, scale))

    # Scale figure size in proportion so on-screen density stays readable.
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.5 * cols * scale, 6.3 * rows * scale),
        squeeze=False,
    )

    coords = {"h": (0.0, 0.0)}
    radius = scale
    for idx, node in enumerate(Nw_sorted):
        ang = 2 * math.pi * idx / n_w - math.pi / 2
        coords[node] = (radius * math.cos(ang), radius * math.sin(ang))

    offset = inst.shift_offset
    for ax_idx, (k, r) in enumerate(used):
        ax = axes[ax_idx // cols][ax_idx % cols]
        colour = _vehicle_colour(inst.K.index(k))
        try:
            seq = sol.route_order(k, r)
        except ValueError as ex:
            ax.text(0.5, 0.5, f"route ({k}, {r}) malformed:\n{ex}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        # Draw all work-centre nodes as faint background
        for node, (px, py) in coords.items():
            if node not in seq:
                ax.plot(px, py, "o", color="lightgrey", markersize=8, zorder=1)
                ax.text(px, py + 0.08 * scale, node, ha="center", va="bottom",
                        color="grey", fontsize=7)

        # Draw visited nodes prominently — node id only.
        # Clock-time labels were removed because they cluttered the diagram
        # and don't scale to instances with many nodes; per-arc timing now
        # lives in result.xlsx's `route_timings` sheet.
        for node in seq:
            px, py = coords[node]
            is_depot = (node == "h")
            ax.plot(px, py, "s" if is_depot else "o",
                    color=colour, markersize=14 if is_depot else 12,
                    markeredgecolor="black", zorder=3)
            ax.text(px, py - 0.12 * scale, node, ha="center", va="top",
                    fontsize=8, fontweight="bold", zorder=4)

        # Compute carrying state on each arc (before drawing arcs so we can label)
        carrying_map = {
            (i, j): aboard for (i, j, aboard) in carrying_on_arcs(sol, inst, k, r)
        }

        # Distances stay on the arcs (white box at the chord midpoint).
        # The yellow carrying box, when non-empty, is placed OUTSIDE the node
        # circle and linked back to the arc midpoint with a thin dotted leader,
        # so it never overlaps with the arrow.
        LABEL_RADIUS = 1.55 * scale      # node circle has radius `scale`
        for s_idx in range(len(seq) - 1):
            a, b = seq[s_idx], seq[s_idx + 1]
            ax_, ay = coords[a]
            bx, by = coords[b]
            ax.annotate(
                "", xy=(bx, by), xytext=(ax_, ay),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2,
                                shrinkA=12, shrinkB=12),
                zorder=2,
            )
            mid_x, mid_y = (ax_ + bx) / 2, (ay + by) / 2

            # Per-arc travel time is no longer drawn on the diagram; the
            # detailed timing/distance breakdown is rendered separately as
            # route_timings.png and as the route_timings sheet of result.xlsx.

            # --- Carrying box outside the circle, with leader ---
            carry = carrying_map.get((a, b), frozenset())
            if not carry:
                continue

            dxv = bx - ax_
            dyv = by - ay
            arc_len = math.hypot(dxv, dyv)
            if arc_len > 1e-6:
                perp_x = -dyv / arc_len
                perp_y = dxv / arc_len
                plus_d = math.hypot(mid_x + perp_x, mid_y + perp_y)
                minus_d = math.hypot(mid_x - perp_x, mid_y - perp_y)
                if minus_d > plus_d:
                    perp_x, perp_y = -perp_x, -perp_y
            else:
                perp_x, perp_y = 0.0, 1.0

            bq = 2.0 * (mid_x * perp_x + mid_y * perp_y)
            cq = mid_x * mid_x + mid_y * mid_y - LABEL_RADIUS * LABEL_RADIUS
            disc = bq * bq - 4.0 * cq
            if disc >= 0:
                t = (-bq + math.sqrt(disc)) / 2.0
                t = max(t, 0.25)
            else:
                t = 0.7
            label_x = mid_x + t * perp_x
            label_y = mid_y + t * perp_y

            ax.plot([mid_x, label_x], [mid_y, label_y],
                    color="#d4a017", linestyle=":", linewidth=0.8,
                    alpha=0.65, zorder=2)

            carry_text = ",".join(sorted(carry))
            ax.text(label_x, label_y, f"[{carry_text}]",
                    ha="center", va="center",
                    fontsize=7, color="#5c4a00", alpha=0.95,
                    bbox=dict(facecolor="#fff8e1", edgecolor="#d4a017",
                              pad=2, alpha=0.95, boxstyle="round,pad=0.3"),
                    zorder=4)

        # List products on this route with O->D, ready, pickup, deliver, wait
        prods = sorted([p for p, (kk, rr) in sol.f_assigned.items()
                        if kk == k and rr == r])

        # One-line totals for the title: travel + service summed along the
        # traversed sequence. The detailed breakdown lives in result.xlsx
        # under the route_timings sheet.
        travel_total = sum(
            inst.c.get((seq[i], seq[i + 1]), 0.0)
            for i in range(len(seq) - 1)
        )
        service_total = sum(
            inst.s_load[p] + inst.s_unload[p] for p in prods
        )
        title_str = (
            f"({k}, R{r})   total {travel_total + service_total:.1f} min "
            f"(travel {travel_total:.1f} + service {service_total:.1f})"
        )
        ax.set_title(title_str, fontsize=10, fontweight="bold",
                     loc="left", pad=8)
        if prods:
            vr_tag = f"({k},R{r})"
            header = (
                f"{'product':<8}{'O→D':<10}{'(V,R)':<10}{'ready':<7}"
                f"{'pickup':<8}{'deliver':<9}{'wait':>6}"
            )
            lines = [header, "-" * len(header)]
            for p in prods:
                op, dp = inst.o[p], inst.d[p]
                ready_s = _hhmm(inst.e[p], offset)
                pickup_t = sol.td.get((op, k, r), 0.0)
                deliver_t = sol.ta.get((dp, k, r), 0.0) + inst.s_unload[p]
                pickup_s = _hhmm(pickup_t, offset)
                deliver_s = _hhmm(deliver_t, offset)
                wait_v = sol.w.get(p, 0.0)
                arrow = f"{op}→{dp}"
                lines.append(
                    f"{p:<8}{arrow:<10}{vr_tag:<10}{ready_s:<7}"
                    f"{pickup_s:<8}{deliver_s:<9}{wait_v:>6.1f}"
                )
            prods_text = "\n".join(lines)
        else:
            prods_text = "(no products on this route)"
        # Render the product list as a left-aligned monospace block below the
        # circle. Slightly smaller font to fit the wider table.
        ax.text(-1.85 * scale, -1.55 * scale, prods_text,
                ha="left", va="top",
                fontfamily="monospace", fontsize=7, color="#222",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc",
                          boxstyle="round,pad=0.3"))
        ax.set_xlim(-1.95 * scale, 1.95 * scale)
        # Extra room top for label boxes, bottom for product table.
        # Vertical extent below the circle is held roughly constant so the
        # table doesn't stretch awkwardly when scale grows.
        ax.set_ylim(-1.95 * scale - 1.10, 1.95 * scale)
        ax.set_aspect("equal")
        ax.axis("off")

    # Hide any empty subplots
    for ax_idx in range(len(used), rows * cols):
        axes[ax_idx // cols][ax_idx % cols].axis("off")

    fig.suptitle(
        f"Routes — case={inst.config['product_set_id']}, "
        f"|P|={len(inst.P)}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Waiting-time bar chart
# =============================================================================


def plot_wait_bars(sol, inst, out_path: Path) -> None:
    """Bar chart of waiting time per product, sorted by ready time."""
    products = sorted(inst.P, key=lambda p: (inst.e[p], p))
    waits = [sol.w[p] for p in products]
    offset = inst.shift_offset

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(10, 0.7 * len(products) + 4), 6),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    bars = ax_top.bar(products, waits, color="#2E74B5", edgecolor="black",
                      linewidth=0.4)
    for bar, w in zip(bars, waits):
        if w > 0.05:
            ax_top.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{w:.1f}", ha="center", va="bottom", fontsize=7)
    ax_top.set_ylabel("Waiting time (min)")
    ax_top.set_title(
        f"Waiting time per product - case={inst.config['product_set_id']}, "
        f"sum wait = {sol.total_wait:.2f} min",
        fontsize=12,
    )
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)

    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_yticks([])
    ax_bottom.set_xlim(ax_top.get_xlim())
    for i, p in enumerate(products):
        ax_bottom.text(i, 0.5, _hhmm(inst.e[p], offset),
                       ha="center", va="center", fontsize=7,
                       color="#555555")
    ax_bottom.set_xticks(range(len(products)))
    ax_bottom.set_xticklabels(products, rotation=45, ha="right", fontsize=8)
    ax_bottom.set_ylabel("ready @", fontsize=8)
    for spine in ("top", "right", "left", "bottom"):
        ax_bottom.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Carrying timeline per vehicle
# =============================================================================
def plot_carrying(sol, inst, out_path: Path) -> None:
    """Per-vehicle carrying chart: one horizontal bar per product, spanning
    [pickup time at o_p, delivery time at d_p]. Overlapping bars indicate
    the vehicle is carrying multiple products simultaneously."""
    used_vehicles = [
        k for k in inst.K
        if any(sol.z_used.get((k, r), False) for r in inst.routes_of(k))
    ]
    offset = inst.shift_offset

    if not used_vehicles:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No vehicles used", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    products_by_vehicle = {}
    for k in used_vehicles:
        items = []
        for p, (kk, rr) in sol.f_assigned.items():
            if kk != k:
                continue
            pickup_t = sol.td[(inst.o[p], k, rr)]
            dropoff_t = sol.ta[(inst.d[p], k, rr)]
            items.append((pickup_t, dropoff_t, p, rr))
        items.sort()
        products_by_vehicle[k] = items

    sorted_products = sorted(inst.P)
    prod_colour = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(sorted_products)}

    fig_h = max(2.0 * len(used_vehicles),
                0.55 * sum(max(1, len(items)) for items in products_by_vehicle.values()) + 2.0)
    fig, axes = plt.subplots(len(used_vehicles), 1,
                             figsize=(13, fig_h),
                             sharex=True, squeeze=False)

    for vi, k in enumerate(used_vehicles):
        ax = axes[vi][0]
        cap = inst.q_k[k]
        items = products_by_vehicle[k]

        if not items:
            ax.text(0.5, 0.5, f"{k}: idle", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="grey")
            ax.set_yticks([])
            continue

        for i_p, (pickup_t, dropoff_t, p, rr) in enumerate(items):
            width = dropoff_t - pickup_t
            ax.barh(i_p, width, left=pickup_t,
                    height=0.7, color=prod_colour[p],
                    edgecolor="black", linewidth=0.5)
            label = (f"{p}  {inst.o[p]}->{inst.d[p]}  "
                     f"q={inst.q_p[p]:.1f}m2  (r={rr})")
            ax.text(pickup_t + max(width * 0.02, 0.3), i_p, label,
                    va="center", ha="left", fontsize=8, color="black",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.6, pad=1))

        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([item[2] for item in items], fontsize=8)
        ax.invert_yaxis()

        total_area = sum(inst.q_p[item[2]] for item in items)
        ax.set_title(
            f"{k} - capacity {cap:.1f} m2, "
            f"{len(items)} product(s), total area {total_area:.1f} m2",
            loc="left", fontsize=10,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[-1][0].set_xlabel("Minutes since shift start")
    ax_top = axes[0][0].twiny()
    ax_top.set_xlim(axes[0][0].get_xlim())
    xticks = axes[0][0].get_xticks()
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels([_hhmm(t, offset) for t in xticks])
    ax_top.set_xlabel("Clock time")

    fig.suptitle(
        f"Carrying timeline - case={inst.config['product_set_id']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Product movement timeline
# =============================================================================
def plot_product_movements(sol, inst, out_path: Path) -> None:
    """Per-product lifecycle on a single timeline.

    For each product, one horizontal row showing three coloured phases:
      1. Ready & waiting at origin     [e_p ... td[o_p, k, r]]   yellow
      2. In transit on the vehicle     [td[o_p] ... ta[d_p]]     vehicle colour
      3. Unloading at destination      [ta[d_p] ... ta[d_p]+s_u] hatched, lighter
    Plus a ready-time marker, origin tag inside the yellow bar, destination
    tag inside the transit bar, and a vehicle ID + total wait on the right.
    """
    products = sorted(inst.P, key=lambda p: inst.e[p])
    offset = inst.shift_offset
    fig, ax = plt.subplots(
        figsize=(15, max(5.0, 0.80 * len(products) + 2.8))
    )
    vehicle_color = {k: _vehicle_colour(i) for i, k in enumerate(inst.K)}

    rightmost = 0.0
    for i_p, p in enumerate(products):
        y = i_p
        if p not in sol.f_assigned:
            ax.text(0, y, f"  {p}: NOT ASSIGNED", va="center", ha="left",
                    fontsize=9, color="red", fontweight="bold")
            continue
        k, r = sol.f_assigned[p]
        op = inst.o[p]
        dp = inst.d[p]
        ep = inst.e[p]
        su = inst.s_unload[p]
        td_op = sol.td[(op, k, r)]
        ta_dp = sol.ta[(dp, k, r)]
        delivery_complete = ta_dp + su
        wait_time = sol.w[p]
        vcolor = vehicle_color[k]

        # Phase 1: ready & waiting (yellow). Always show the origin tag;
        # place it INSIDE the bar when wide enough, otherwise just ABOVE the
        # row so it stays readable for narrow bars or zero-wait products.
        wait_span = td_op - ep
        BAR_H = 0.5
        WAIT_INSIDE_MIN = 12.0
        if wait_span > 0.5:
            ax.barh(y, wait_span, left=ep, height=BAR_H,
                    color="#fff3cd", edgecolor="#d4a017", linewidth=0.6)
        origin_anchor_x = ep + max(wait_span, 0.0) / 2.0
        if wait_span >= WAIT_INSIDE_MIN:
            ax.text(origin_anchor_x, y, f"@{op}",
                    va="center", ha="center",
                    fontsize=7, color="#5c4a00", fontweight="bold")
        else:
            ax.text(origin_anchor_x, y - 0.34, f"@{op}",
                    va="bottom", ha="center",
                    fontsize=7, color="#5c4a00", fontweight="bold")

        # Ready-time marker. Clock stamp placed BELOW the bar (below the row)
        # so it never collides with the @origin tag above narrow bars.
        ax.plot(ep, y, marker="o", color="#d4a017", markersize=6,
                markeredgecolor="black", zorder=4)
        ax.text(ep, y + 0.30, _hhmm(ep, offset),
                va="top", ha="center",
                fontsize=6, color="#5c4a00")

        # Phase 2: in transit (vehicle colour). Same inside-vs-above logic.
        transit_span = ta_dp - td_op
        TRANSIT_INSIDE_MIN = 8.0
        ax.barh(y, transit_span, left=td_op, height=BAR_H,
                color=vcolor, alpha=0.85,
                edgecolor="black", linewidth=0.4)
        dest_anchor_x = td_op + transit_span / 2.0
        if transit_span >= TRANSIT_INSIDE_MIN:
            ax.text(dest_anchor_x, y, f"->{dp}",
                    va="center", ha="center",
                    fontsize=8, color="white", fontweight="bold")
        elif transit_span > 0.5:
            ax.text(dest_anchor_x, y - 0.34, f"->{dp}",
                    va="bottom", ha="center",
                    fontsize=7, color=vcolor, fontweight="bold")

        ax.barh(y, max(su, 0.4), left=ta_dp, height=BAR_H,
                color=vcolor, alpha=0.40,
                edgecolor="black", linewidth=0.3, hatch="//")

        right_label = f"  ({k},R{r})    wait={wait_time:.1f} min"
        ax.text(delivery_complete + 1.5, y, right_label,
                va="center", ha="left", fontsize=8)
        rightmost = max(rightmost, delivery_complete + 40)

    ax.set_yticks(range(len(products)))
    ax.set_yticklabels(products, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Minutes since shift start")
    ax.set_xlim(left=-2, right=max(rightmost, inst.T_max * 0.3))
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    xticks = ax.get_xticks()
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels([_hhmm(t, offset) for t in xticks])
    ax_top.set_xlabel("Clock time")

    legend_handles = [
        mpatches.Patch(facecolor="#fff3cd", edgecolor="#d4a017",
                       label="ready & waiting at origin"),
        mpatches.Patch(facecolor="grey", alpha=0.85,
                       label="in transit (colour = vehicle)"),
        mpatches.Patch(facecolor="grey", alpha=0.40, hatch="//",
                       label="unloading at destination"),
        plt.Line2D([0], [0], marker="o", color="#d4a017", markersize=7,
                   markeredgecolor="black", linestyle="",
                   label="ready time e_p"),
    ]
    for vi, k in enumerate(inst.K):
        legend_handles.append(
            mpatches.Patch(facecolor=_vehicle_colour(vi), alpha=0.85,
                           label=f"vehicle {k}")
        )
    ax.legend(handles=legend_handles, loc="upper right",
              framealpha=0.95, fontsize=7, ncol=2)

    ax.set_title(
        f"Product movements - case={inst.config['product_set_id']}, "
        f"sum wait = {sol.total_wait:.2f} min",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Pareto frontier plot (multi-objective sweep)
# =============================================================================


def plot_pareto_frontier(rows, out_path: Path, *,
                         x_label: str = "Route duration (min)",
                         y_label: str = "Total waiting time (min)",
                         title: str = "Pareto frontier") -> None:
    """Plot the Pareto frontier from an iteration log.

    rows is a list of dicts with keys 'route_duration_min' and
    'total_wait_min'. Infeasible iterations are silently skipped.
    Points are drawn as a scatter (no connecting line). Each point is
    annotated with its (x, y) value.
    """
    pts = [(r["route_duration_min"], r["total_wait_min"], r.get("iter"))
           for r in rows
           if r.get("route_duration_min") is not None
           and r.get("total_wait_min") is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    if not pts:
        ax.text(0.5, 0.5, "no feasible Pareto points",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="grey")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    pts.sort(key=lambda t: (t[0], t[1]))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    iters = [p[2] for p in pts]

    ax.scatter(xs, ys, s=70, color="#1F3864",
               edgecolors="black", linewidths=0.8, zorder=3)

    for x, y, i in zip(xs, ys, iters):
        label = f"({x:.1f}, {y:.1f})"
        if i is not None:
            label = f"#{i}: {label}"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(7, 5), textcoords="offset points",
            fontsize=8, color="#1F3864",
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{title}  ({len(pts)} points)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Convenience: render all five at once
# =============================================================================
def plot_route_timings(sol, inst, out_path: Path) -> None:
    """Render a detail page combining per-(k,r) timing tables and the
    active-node distance matrix.

    The page has two sections:
      1. One text table per used (vehicle, route) listing every traversed
         arc with depart/arrive clock times, travel time, and the
         loading/unloading service at the endpoints.
      2. The full active-node distance matrix (travel times between every
         pair of nodes in inst.N), for reference.
    """
    offset = inst.shift_offset

    # Collect per-(k, r) timing data first.
    timing_blocks = []  # list of (title_str, list_of_lines)
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue
            prods_on_route = [
                p for p, (kk, rr) in sol.f_assigned.items()
                if kk == k and rr == r
            ]
            title = f"({k}, R{r})   |  products: {', '.join(sorted(prods_on_route)) or '—'}"
            header = (
                f"{'leg':<4}{'from':<8}{'to':<8}"
                f"{'depart':<8}{'arrive':<8}"
                f"{'travel':>8}{'load@from':>11}{'unload@to':>11}"
            )
            lines = [header, "-" * len(header)]
            travel_sum = 0.0
            service_sum = 0.0
            for leg, (i_node, j_node) in enumerate(
                    zip(seq[:-1], seq[1:]), start=1):
                depart_i = sol.td.get((i_node, k, r), 0.0)
                arrive_j = sol.ta.get((j_node, k, r), 0.0)
                travel = inst.c.get((i_node, j_node), 0.0)
                load_at_i = sum(
                    inst.s_load[p] for p in inst.P
                    if inst.o[p] == i_node
                    and sol.f_assigned.get(p) == (k, r)
                )
                unload_at_j = sum(
                    inst.s_unload[p] for p in inst.P
                    if inst.d[p] == j_node
                    and sol.f_assigned.get(p) == (k, r)
                )
                travel_sum += travel
                service_sum += load_at_i + unload_at_j
                lines.append(
                    f"{leg:<4}{i_node:<8}{j_node:<8}"
                    f"{_hhmm(depart_i, offset):<8}{_hhmm(arrive_j, offset):<8}"
                    f"{travel:>8.1f}{load_at_i:>11.2f}{unload_at_j:>11.2f}"
                )
            lines.append("-" * len(header))
            lines.append(
                f"{'TOTAL':<4}{'':<8}{'':<8}{'':<8}{'':<8}"
                f"{travel_sum:>8.1f}{service_sum:>11.2f}{'':>11}"
                f"     (travel + service = {travel_sum + service_sum:.1f})"
            )
            timing_blocks.append((title, lines))

    # ---- Distance matrix block ----
    nodes_for_matrix = list(inst.N)
    n = len(nodes_for_matrix)
    col_w = max(5, max(len(s) for s in nodes_for_matrix) + 1)
    dist_lines = []
    header_row = " " * col_w + "".join(f"{j:>{col_w}}" for j in nodes_for_matrix)
    dist_lines.append(header_row)
    dist_lines.append("-" * len(header_row))
    for i in nodes_for_matrix:
        cells = []
        for j in nodes_for_matrix:
            if i == j:
                cells.append("—")
            else:
                v = inst.c.get((i, j))
                cells.append(f"{v:.0f}" if v is not None else "·")
        dist_lines.append(f"{i:>{col_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    # ---- Layout: timing tables on the left, distance matrix on the right ----
    n_routes = max(1, len(timing_blocks))
    width = 13 + max(0.45 * n, 4.0)
    height = max(8.0, 1.6 + 0.45 * sum(len(b[1]) + 2 for b in timing_blocks)
                 + 0.4 * n)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [3, max(2, 0.15 * n + 1)]},
    )

    # Timing tables column
    ax_left.set_axis_off()
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    if not timing_blocks:
        ax_left.text(0.5, 0.5, "No routes used.",
                     ha="center", va="center", fontsize=11, color="grey")
    else:
        total_lines = sum(len(b[1]) + 2 for b in timing_blocks)
        y_step = 0.95 / max(total_lines, 1)
        y = 0.97
        for title, lines in timing_blocks:
            ax_left.text(0.0, y, title, fontfamily="monospace",
                         fontsize=9, fontweight="bold", color="#1F3864",
                         va="top")
            y -= y_step
            for line in lines:
                ax_left.text(0.0, y, line, fontfamily="monospace",
                             fontsize=8, color="#222", va="top")
                y -= y_step
            y -= y_step  # blank line between blocks

    # Distance matrix column
    ax_right.set_axis_off()
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.text(0.5, 0.99, "Distance / travel-time matrix (minutes)",
                  fontfamily="monospace", fontsize=9, fontweight="bold",
                  color="#1F3864", ha="center", va="top")
    y = 0.95
    y_step_r = 0.92 / max(len(dist_lines), 1)
    for line in dist_lines:
        ax_right.text(0.0, y, line, fontfamily="monospace",
                      fontsize=7, color="#222", va="top")
        y -= y_step_r

    fig.suptitle(
        f"Route timings — case={inst.config['product_set_id']}, "
        f"|P|={len(inst.P)}, |K|={len(inst.K)}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_all(sol, inst, output_dir: Path, suffix: str = "",
                prefix: str = "") -> dict:
    """Render Gantt, route diagram, wait bars, carrying, product_movements,
    and the route_timings detail page.

    The `suffix` argument appends an underscore-separated tag (typically the
    run timestamp) to every PNG filename so artefacts inherit the parent
    folder's timestamp. `prefix` is retained for backward compatibility but
    is rarely needed alongside `suffix`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""
    sfx = f"_{suffix}" if suffix else ""
    paths = {
        "gantt":              output_dir / f"{pfx}gantt{sfx}.png",
        "routes":             output_dir / f"{pfx}routes{sfx}.png",
        "wait":               output_dir / f"{pfx}wait{sfx}.png",
        "carrying":           output_dir / f"{pfx}carrying{sfx}.png",
        "product_movements":  output_dir / f"{pfx}product_movements{sfx}.png",
        "route_timings":      output_dir / f"{pfx}route_timings{sfx}.png",
    }
    plot_gantt(sol, inst, paths["gantt"])
    plot_route_diagram(sol, inst, paths["routes"])
    plot_wait_bars(sol, inst, paths["wait"])
    plot_carrying(sol, inst, paths["carrying"])
    plot_product_movements(sol, inst, paths["product_movements"])
    plot_route_timings(sol, inst, paths["route_timings"])
    return paths
