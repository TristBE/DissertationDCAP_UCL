
import os
import sys
import matplotlib
matplotlib.use('Agg')              
import matplotlib.pyplot as plt
import numpy as np

from dcABP_circular_wall_analysis_FINAL import (
    dcABPCircularWall,
    parameter_sweep_omega,
    parameter_sweep_R_w,
    parameter_sweep_k,
    bifurcation_diagram_omega,
    bifurcation_diagram_R_w,
)
from dcABP_circular_wall_playground_FINAL import (
    _plot_phase_portrait_with_density,
    _plot_polar_two_views,
    _plot_cartesian_trajectories,
    _plot_cartesian_trajectory_improved,
    _plot_trajectories_on_phase_portrait_with_density,
)


# =============================================================================
#                  GLOBAL DEFAULTS (override per-run as needed)
# =============================================================================

GLOBAL_DEFAULTS = {
    "dpi": 150,

    # Cartesian phase portrait + trajectory overlay
    "cart_resolution": 50,
    "cart_density": 2.5,
    "cart_figsize": (10, 8),

    # Polar two-view
    "polar_full_r_range": None,        # None = auto from buffer/padding/width
    "polar_zoom_r_range": None,
    "polar_inner_buffer": 0.05,
    "polar_outer_padding": 0.5,
    "polar_wall_zoom_width": 3.0,
    "polar_resolution": 200,
    "polar_density": 3.0,
    "polar_figsize": (16, 8),          # auto-bumped to (18, 9) when R_w < 1

    # Multi-trajectory Cartesian
    "t_span": (0, 165),
    "traj_zoom_padding": 1.5,
    "traj_wall_depth": 2.0,
    "traj_figsize": (12, 12),

    # Phase-space trajectory overlay
    "phase_traj_t_span": (0, 165),
    "phase_traj_resolution": 40,
    "phase_traj_density": 2.0,
    "phase_traj_figsize": (10, 8),
}


# =============================================================================
#                                  RUNS
# =============================================================================

def _zoom_for(R_w):
    if R_w >= 8:
        return 3.0
    elif R_w >= 3:
        return 2.0
    elif R_w >= 1:
        return min(1.0, 0.6 * R_w)
    else:
        return 0.5 * R_w


PLOTS = {"polar", "phase_traj", "trajectories"}

RUNS = [
    # ----- 1.1 -----
    {
        "name": "1.1_Outside",
        "omega": -0.5, "R_w": 10.0, "k": 0.5, "confinement": "outside",
        "plots": PLOTS,
        "ics": [
            (12.0, 0.5, 0.0),     # bulk, far from wall
            (10.0, 3.14, 1.57),   # at the wall, started elsewhere
            (10.5, 2.0, 3.14),    # close to wall, different phi
            (5.0, 4.5, 4.71),     # well inside the wall force region
        ],
        "polar_wall_zoom_width": 3.0,
        "polar_outer_padding": 1.0,
    },
    {
        "name": "1.1_Inside",
        "omega": -0.5, "R_w": 10.0, "k": 0.5, "confinement": "inside",
        "plots": PLOTS,
        "ics": [
            (5.0, 3.14, 0.0),
            (10.0, 2.8, 0.0),
            (11.0, 4.8, 0.0),
            (12.0, 1.7, 0.0),
        ],
        "polar_wall_zoom_width": 3.0,
        "polar_outer_padding": 1.0,
    },

    # ----- 1.2 -----
    {
        "name": "1.2_Inside",
        "omega": 0.02, "R_w": 1.5, "k": 20.0, "confinement": "inside",
        "plots": PLOTS,
        "ics": [
            (0.3, 0.5, 0.0),
            (0.7, 2.0, 0.0),
            (1.0, 4.0, 0.0),
            (1.4, 5.5, 0.0),
        ],
        "polar_wall_zoom_width": 0.6,
        "polar_outer_padding": 0.2,
    },
    {
        "name": "1.2_Outside",
        "omega": 0.02, "R_w": 1.5, "k": 20.0, "confinement": "outside",
        "plots": PLOTS,
        "ics": [
            (0.0, 3.14, 0.0),
            (1.0, 2.8, 0.0),
            (2.0, 4.8, 0.0),
            (2.5, 1.7, 0.0),
        ],
        "polar_wall_zoom_width": 0.4,
        "polar_outer_padding": 0.12,
    },

    # ----- 2.1 -----
    {
        "name": "2.1_Inside",
        "omega": -1.1, "R_w": 2.22, "k": 10.0, "confinement": "inside",
        "plots": PLOTS,
        "ics": [
            (0.0, 3.14, 0.0),
            (1.0, 2.8, 0.0),
            (2.0, 4.8, 0.0),
            (4.0, 1.7, 0.0),
        ],
        "polar_wall_zoom_width": 0.6,
        "polar_outer_padding": 0.18,
    },
    {
        "name": "2.1_Outside",
        "omega": -1.1, "R_w": 2.22, "k": 20.0, "confinement": "outside",
        "plots": PLOTS,
        "ics": [
            (0.0, 1.0, 0.0),
            (4.0, 3.14, 1.5),
            (2.5, 5.0, 3.0),
            (2.5, 2.0, 4.5),
        ],
        "polar_wall_zoom_width": 1.4,
        "polar_outer_padding": 0.3,
    },

    # ----- 2.2 -----
    {
        "name": "2.2_Outside",
        "omega":  1.0, "R_w": 0.32, "k": 20.0, "confinement": "outside",
        "plots": PLOTS,
        "ics": [
            (0.6, 1.0, 0.0),
            (1.0, 3.14, 1.5),
            (0.8, 5.0, 3.0),
            (1.2, 2.0, 4.5),
        ],
        "polar_wall_zoom_width": 0.22,
        "polar_outer_padding": 0.08,
    },
    {
        "name": "2.2_Inside",
        "omega": 1.0, "R_w": 0.32, "k": 20.0, "confinement": "inside",
        "plots": PLOTS,
        "ics": [
            (0.0, 3.14, 0.0),
            (0.2, 2.8, 0.0),
            (0.4, 4.8, 0.0),
            (0.6, 1.7, 0.0),
        ],
        "polar_wall_zoom_width": 0.15,
        "polar_outer_padding": 0.05,
    },
]


# =============================================================================
#                              RUNNER LOGIC
# =============================================================================

OVERWRITE_EXISTING = True   # If False, skip runs whose folder already exists


def get_param(run, key):
    return run.get(key, GLOBAL_DEFAULTS.get(key))


def save_fig(fig, folder, name, dpi):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved {path}")


def check_duplicate_names(runs):
    seen, dupes = set(), []
    for r in runs:
        n = r["name"]
        if n in seen:
            dupes.append(n)
        seen.add(n)
    if dupes:
        print(f"WARNING: duplicate names in RUNS: {dupes}")
        print("Later entries will overwrite earlier ones.\n")


def run_one(run):
    # Defensive: close any leftover figures before this run
    plt.close('all')

    name = run["name"]
    omega = run["omega"]
    R_w = run["R_w"]
    k = run["k"]
    confinement = run["confinement"]
    plots = set(run.get("plots", {"polar", "phase_traj", "trajectories"}))
    dpi = get_param(run, "dpi")

    folder = name
    if os.path.exists(folder) and not OVERWRITE_EXISTING:
        print(f"\n[{name}]  folder exists -- SKIPPED "
              "(set OVERWRITE_EXISTING=True to re-run)")
        return
    os.makedirs(folder, exist_ok=True)

    print(f"\n[{name}]  omega={omega}, R_w={R_w}, k={k}, {confinement}")
    print(f"    plots: {sorted(plots)}")

    system = dcABPCircularWall(
        omega_tilde=omega, R_w_tilde=R_w,
        k_tilde=k, confinement=confinement,
    )

    if confinement == 'outside' and run.get("ics"):
        from dcABP_circular_wall_analysis_FINAL import (
            find_fixed_points, stability_analysis,
        )
        fps = find_fixed_points(system)
        if fps:
            results = stability_analysis(system, verbose=False)
            # Prefer a stable fixed point; otherwise use the first one
            stable_idx = next(
                (i for i, r in enumerate(results)
                 if r['stability'] == 'Stable'), None)
            idx = stable_idx if stable_idx is not None else 0
            r_fp, theta_fp = fps[idx]
            fp_ic = (r_fp, theta_fp, 0.0)
            # Replace original ics with FP-anchored + originals
            original_ics = run["ics"]
            if not any(abs(ic[0] - r_fp) < 1e-6 and abs(ic[1] - theta_fp) < 1e-6
                       for ic in original_ics):
                run = {**run, "ics": [fp_ic] + list(original_ics)}
                print(f"    prepended fixed-point IC: "
                      f"({r_fp:.3f}, {theta_fp:.3f}, 0.0)")

    if "polar" in plots:
        polar_figsize = get_param(run, "polar_figsize")
        if R_w < 1.0:
            polar_figsize = (18, 9)
        fig = _plot_polar_two_views(
            system,
            full_r_range=get_param(run, "polar_full_r_range"),
            zoom_r_range=get_param(run, "polar_zoom_r_range"),
            inner_buffer=get_param(run, "polar_inner_buffer"),
            outer_padding=get_param(run, "polar_outer_padding"),
            wall_zoom_width=get_param(run, "polar_wall_zoom_width"),
            resolution=get_param(run, "polar_resolution"),
            density=get_param(run, "polar_density"),
            figsize=polar_figsize,
            show_fixed_points=True,
        )
        save_fig(fig, folder, "phase_portrait_polar.png", dpi)

    if "phase_traj" in plots:
        ics = run.get("ics")
        if not ics:
            print("    [skip] 'phase_traj' requires 'ics'")
        else:
            ics_2d = [(ic[0], ic[1]) for ic in ics]
            fig = _plot_trajectories_on_phase_portrait_with_density(
                system, ics_2d,
                t_span=get_param(run, "phase_traj_t_span"),
                resolution=get_param(run, "phase_traj_resolution"),
                density=get_param(run, "phase_traj_density"),
                figsize=get_param(run, "phase_traj_figsize"),
            )
            save_fig(fig, folder, "phase_portrait_with_trajectories.png", dpi)

    if "trajectories" in plots:
        ics = run.get("ics")
        if not ics:
            print("    [skip] 'trajectories' requires 'ics'")
        else:
            base_padding = get_param(run, "traj_zoom_padding")
            max_r0 = max(ic[0] for ic in ics)
            needed = max_r0 - R_w + 0.5
            zoom_padding = max(base_padding, needed)
            fig = _plot_cartesian_trajectories(
                system, ics,
                t_span=get_param(run, "t_span"),
                zoom_padding=zoom_padding,
                wall_depth=get_param(run, "traj_wall_depth"),
                figsize=get_param(run, "traj_figsize"),
            )
            save_fig(fig, folder, "trajectories_cartesian.png", dpi)


def main():
    print(f"Running {len(RUNS)} configurations\n")
    check_duplicate_names(RUNS)
    for run in RUNS:
        try:
            run_one(run)
        except Exception as e:
            print(f"    ERROR in {run.get('name', '?')}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
