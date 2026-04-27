"""
Doubly Chiral Active Brownian Particles (dcABP) - Circular Wall Playground (Complete)
======================================================================================


HOW TO USE:
-----------
1. Set the OUTPUT_FOLDER name for your run
2. Adjust the SYSTEM PARAMETERS (omega~, R~_w, k~, confinement)
3. Set Y/N toggles (True/False) for which analyses you want
4. Adjust any specific settings for enabled analyses
5. Run: python dcABP_circular_wall_playground_FINAL.py

All adjustable parameters are at the top of the file for easy modification.
Results are automatically saved to your custom-named folder.

FEATURES AVAILABLE:
-------------------
- Core: Stability analysis, Cartesian phase portraits, nullclines
- Polar phase portraits: Streamlines (bullseye), quiver plots
- Parameter sweeps: omega~, R~_w, k~ sweeps
- Bifurcation diagrams: vs omega~, vs R~_w
- Trajectory integration (phase space and Cartesian)
- Cartesian multi-trajectory plotting (all ICs on one x,y plot)
- Inside vs Outside confinement comparison
- Basin of attraction analysis (supervisor question 1)
- sin(theta*) systematic analysis (supervisor question 2)
- Two-parameter phase diagram in (omega~, 1/R~_w) space (supervisor question 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from dcABP_circular_wall_analysis_FINAL import *
except ImportError:
    print("ERROR: Could not import dcABP_circular_wall_analysis_FINAL")
    print("Make sure the master file is in the same directory.")
    sys.exit(1)


OUTPUT_FOLDER = "2.2_Outside"   # Change this for each new run!

# -----------------------------------------------------------------------------
# MAIN SYSTEM PARAMETERS
# -----------------------------------------------------------------------------
OMEGA_TILDE = 1.0       # Rescaled intrinsic frequency: omega~ = omega/(alpha*v)
R_W_TILDE = 0.32            # Rescaled wall radius: R~_w = alpha * R_w
K_TILDE = 10.0               # Rescaled wall stiffness: k~ = mu*k/(alpha*v)
CONFINEMENT = 'outside'      # 'inside' or 'outside'

# -----------------------------------------------------------------------------
# PHASE PORTRAIT SETTINGS
# -----------------------------------------------------------------------------
R_RANGE = (0.0, 6.0)        # Range of r~ values (adjust based on R~_w)
THETA_RANGE = (0, 2*np.pi)  # Range of theta values
RESOLUTION = 50             # Grid resolution for streamplot (higher = denser grid)
DENSITY = 2.5               # Streamplot line density (higher = more streamlines)

RUN_STABILITY_ANALYSIS = True       # Print stability analysis to console
RUN_PHASE_PORTRAIT = True           # Main Cartesian (r~, theta) phase portrait
SHOW_NULLCLINES = False              # Show nullclines on Cartesian phase portrait
SHOW_FIXED_POINTS = True            # Show fixed points on phase portraits

# --- Polar Phase Portraits (bullseye plots) ---
RUN_POLAR_STREAMLINES = True       # Polar plot with integrated streamlines
RUN_POLAR_QUIVER = False            # Polar plot with vector arrows
RUN_CARTESIAN_VS_POLAR = False      # Side-by-side Cartesian vs Polar comparison
POLAR_NEAR_WALL_ZOOM = True         # Zoom to near-wall region in polar plots

# --- Parameter Sweeps ---
RUN_OMEGA_SWEEP = False             # Sweep over omega~ values
RUN_R_W_SWEEP = False               # Sweep over R~_w values
RUN_K_SWEEP = False                 # Sweep over k~ values

# --- Bifurcation Diagrams ---
RUN_BIFURCATION_OMEGA = True       # Bifurcation diagram vs omega~
RUN_BIFURCATION_R_W = True         # Bifurcation diagram vs R~_w

# --- Trajectory Analysis ---
RUN_TRAJECTORIES = True            # Integrate and plot trajectories on phase portrait
RUN_CARTESIAN_TRAJECTORY = True     # Plot single trajectory in (x, y) plane
RUN_CARTESIAN_TRAJECTORIES = True   # Plot ALL trajectory ICs in (x, y) plane

# --- Confinement Comparison ---
RUN_INSIDE_OUTSIDE_COMPARISON = False  # Compare inside vs outside confinement

# --- SUPERVISOR ANALYSIS (Advanced) ---
RUN_BASIN_OF_ATTRACTION = True     # Basin of attraction analysis
RUN_SIN_THETA_ANALYSIS = True      # Systematic sin(theta*) analysis
RUN_PHASE_DIAGRAM = True           # Two-parameter phase diagram

OMEGA_SWEEP_VALUES = [-0.9, -0.5, -0.2, -0.05]
R_W_SWEEP_VALUES = [1.0, 2.0, 5.0, 20.0]
K_SWEEP_VALUES = [1.0, 5.0, 10.0, 50.0]

# --- Bifurcation Diagram Settings ---
OMEGA_BIFURCATION_RANGE = (-0.99, 0.01)
R_W_BIFURCATION_RANGE = (0.1, 25.0)
BIFURCATION_POINTS = 100


# --- Trajectory Settings ---
TRAJECTORY_INITIAL_CONDITIONS = [
    (0.2, 3.14),  #pi/2   # (r0, theta0)
    (0.32, 2.8),  #pi
    (0.4, 1.4),  #pi*2
    (0.5, 1.4),  #
]
TRAJECTORY_TIME_SPAN = (0, 500)
CARTESIAN_TRAJECTORY_IC = (0.45, 0.0, 0.0)  # (r0, theta0, phi0) for single trajectory
CARTESIAN_TRAJECTORY_TIME = (0, 500)

# --- Cartesian Multi-Trajectory Settings ---
CARTESIAN_ZOOM_PADDING = 1.5   # Extra radial distance beyond wall to show
CARTESIAN_WALL_DEPTH = 2.0     # Visual thickness of blue wall shading
CARTESIAN_FIGSIZE = (12, 12)   # Figure size for Cartesian trajectory plots

# --- Polar Phase Portrait Settings ---
POLAR_N_THETA_LINES = 100
POLAR_N_R_LINES = 15
POLAR_QUIVER_RESOLUTION = 100
POLAR_T_MAX = 30

# --- Basin of Attraction Settings ---
BASIN_N_R = 40
BASIN_N_THETA = 40
BASIN_T_MAX = 200

# --- sin(theta*) Analysis Settings ---
SIN_ANALYSIS_N_OMEGA = 50
SIN_ANALYSIS_N_RW = 50
SIN_ANALYSIS_OMEGA_RANGE = (-0.99, 0.01)
SIN_ANALYSIS_RW_RANGE = (0.5, 20.0)

# --- Phase Diagram Settings ---
PHASE_DIAGRAM_N_OMEGA = 100
PHASE_DIAGRAM_N_INV_RW = 100
PHASE_DIAGRAM_OMEGA_RANGE = (-1.5, 4.0)
PHASE_DIAGRAM_INV_RW_RANGE = (0.0, 4.0)

# --- Figure Settings ---
FIGURE_DPI = 150
MAIN_FIGSIZE = (10, 8)
PANEL_FIGSIZE = (15, 10)
COMPARISON_FIGSIZE = (14, 6)
POLAR_FIGSIZE = (10, 10)
CARTESIAN_VS_POLAR_FIGSIZE = (16, 7)
BASIN_FIGSIZE = (14, 6)
SIN_ANALYSIS_FIGSIZE = (16, 12)
PHASE_DIAGRAM_FIGSIZE = (14, 10)

from scipy.integrate import solve_ivp


def _plot_phase_portrait_with_density(system, r_range=None, theta_range=(0, 2*np.pi),
                                      resolution=25, density=1.5,
                                      figsize=(10, 8), show_nullclines=True,
                                      show_fixed_points=True, title=None):
    """
    Phase portrait with controllable streamplot density parameter.

    Identical to the imported plot_phase_portrait but with density as a
    proper parameter instead of hardcoded 1.5.
    """
    R_w = system.R_w_tilde

    if r_range is None:
        if system.confinement == 'inside':
            r_range = (max(0.5, R_w - 1), R_w + 2)
        else:
            r_range = (max(0.5, R_w - 2), R_w + 1)

    r = np.linspace(r_range[0], r_range[1], resolution)
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    R, THETA = np.meshgrid(r, theta)

    dR, dTHETA = system.vector_field(R, THETA)
    speed = np.sqrt(dR**2 + dTHETA**2)

    fig, ax = plt.subplots(figsize=figsize)

    strm = ax.streamplot(R, THETA, dR, dTHETA,
                         color=speed, cmap='viridis',
                         density=density, linewidth=0.8, arrowsize=1.2)
    fig.colorbar(strm.lines, ax=ax, label='|velocity|')

    # Wall
    ax.axvline(x=R_w, color='blue', linewidth=2, linestyle='-',
               label=f'Wall (r = {R_w:.1f})')
    if system.confinement == 'inside':
        ax.axvspan(R_w, r_range[1], alpha=0.15, color='blue', label='Wall region')
    else:
        ax.axvspan(r_range[0], R_w, alpha=0.15, color='blue', label='Wall region')

    # Nullclines
    if show_nullclines:
        theta_fine = np.linspace(theta_range[0], theta_range[1], 500)
        r_null = compute_r_nullcline(system, theta_fine)

        if system.confinement == 'inside':
            mask = (r_null >= R_w) & (r_null <= r_range[1])
        else:
            mask = (r_null >= r_range[0]) & (r_null <= R_w)

        ax.plot(r_null[mask], theta_fine[mask], 'r-', linewidth=2,
                label=r'$\dot{r}=0$ nullcline')

        r_fine = np.linspace(r_range[0], r_range[1], 100)
        theta_null_1, theta_null_2 = compute_theta_nullcline(system, r_fine)

        valid_1 = ~np.isnan(theta_null_1)
        valid_2 = ~np.isnan(theta_null_2)

        if np.any(valid_1):
            ax.plot(r_fine[valid_1], theta_null_1[valid_1], 'b-', linewidth=2,
                    label=r'$\dot{\theta}=0$ nullcline')
        if np.any(valid_2):
            ax.plot(r_fine[valid_2], theta_null_2[valid_2], 'b-', linewidth=2)

    # Fixed points
    if show_fixed_points:
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)

        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                marker, color = 'o', 'green'
            elif 'Saddle' in result['classification']:
                marker, color = 'x', 'red'
            else:
                marker, color = 's', 'orange'

            ax.plot(r_fp, theta_fp, marker, markersize=8,
                    markeredgewidth=2, color=color, alpha=0.6,
                    label=f"{result['classification']} ({r_fp:.2f}, {theta_fp:.2f})")

    ax.set_xlabel(r'$\tilde{r}$ (dimensionless radial position)', fontsize=12)
    ax.set_ylabel(r'$\theta$ (orientation relative to radial, rad)', fontsize=12)
    ax.set_xlim(r_range)
    ax.set_ylim(theta_range)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    if title is None:
        title = (f'Phase Portrait ({system.confinement} confinement)\n'
                 f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, '
                 f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f}, '
                 f'$\\tilde{{k}}$ = {system.k_tilde:.1f}')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_trajectories_on_phase_portrait_with_density(system, initial_conditions,
                                                       r_range=None, t_span=(0, 30),
                                                       resolution=40, density=2.0,
                                                       figsize=(10, 8)):
    """
    Trajectories overlaid on phase portrait with controllable streamline
    resolution and density.

    Accepts (r0, theta0) pairs — phi0=0 is appended automatically for
    integration, then only r and theta are plotted.
    """
    fig = _plot_phase_portrait_with_density(system, r_range=r_range,
                                            resolution=resolution, density=density,
                                            figsize=figsize, show_nullclines=False)
    ax = fig.axes[0]

    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))

    for ic, color in zip(initial_conditions, colors):
        # Convert (r0, theta0) to (r0, theta0, phi0=0) for integrate_full_trajectory
        ic_full = (ic[0], ic[1], 0.0)
        t, r, theta, phi = integrate_full_trajectory(system, ic_full, t_span)

        # Trajectory line (zorder=3 keeps it above wall shading)
        ax.plot(r, theta, '-', color=color, linewidth=2, alpha=0.8, zorder=3)

        # Direction arrows along trajectory
        n_arrows = 6
        arrow_indices = np.linspace(0, len(r) - 2, n_arrows, dtype=int)
        for idx in arrow_indices:
            dr = r[idx + 1] - r[idx]
            dth = theta[idx + 1] - theta[idx]
            if abs(dr) > 1e-12 or abs(dth) > 1e-12:
                ax.annotate('', xy=(r[idx + 1], theta[idx + 1]),
                            xytext=(r[idx], theta[idx]),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.5, mutation_scale=15),
                            zorder=4)

        ax.plot(r[0], theta[0], 'o', color=color, markersize=10,
                label=f'IC: ({ic[0]:.2f}, {ic[1]:.2f})', zorder=5)
        ax.plot(r[-1], theta[-1], 's', color=color, markersize=8, zorder=5)

    ax.legend(loc='upper right', fontsize=8)
    return fig


def _plot_cartesian_trajectory_improved(system, initial_condition,
                                         t_span=(0, 100),
                                         zoom_padding=1.5, wall_depth=2.0,
                                         figsize=(12, 12)):
    """
    Single Cartesian trajectory with blue annular wall shading matching
    phase portrait styling, zoomed view, and thinner trajectory line.
    """
    t, r, theta, phi = integrate_full_trajectory(system, initial_condition, t_span)
    x = r * np.cos(phi)
    y_pos = r * np.sin(phi)

    fig, ax = plt.subplots(figsize=figsize)
    R_w = system.R_w_tilde

    # Wall shading (annular fill matching phase portrait axvspan)
    n_ring = 200
    wall_theta = np.linspace(0, 2 * np.pi, n_ring)

    if system.confinement == 'inside':
        r_inner, r_outer = R_w, R_w + wall_depth
    else:
        r_inner, r_outer = max(0, R_w - wall_depth), R_w

    theta_fill = np.linspace(0, 2 * np.pi, n_ring)
    x_inner = r_inner * np.cos(theta_fill)
    y_inner = r_inner * np.sin(theta_fill)
    x_outer = r_outer * np.cos(theta_fill)
    y_outer = r_outer * np.sin(theta_fill)

    ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
            np.concatenate([y_outer, y_inner[::-1]]),
            color='blue', alpha=0.15, label='Wall region')
    ax.plot(R_w * np.cos(wall_theta), R_w * np.sin(wall_theta),
            color='blue', linewidth=2,
            label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.1f})')

    # Trajectory (zorder=3 keeps it above the blue wall shading)
    ax.plot(x, y_pos, 'k-', linewidth=0.6, alpha=0.7, zorder=3)

    # Direction arrows along trajectory
    n_arrows = 8  # number of arrows along the path
    arrow_indices = np.linspace(0, len(x) - 2, n_arrows, dtype=int)
    for idx in arrow_indices:
        dx = x[idx + 1] - x[idx]
        dy = y_pos[idx + 1] - y_pos[idx]
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            ax.annotate('', xy=(x[idx + 1], y_pos[idx + 1]),
                        xytext=(x[idx], y_pos[idx]),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=1.2, mutation_scale=12),
                        zorder=4)

    ax.plot(x[0], y_pos[0], 'o', color='green', markersize=10,
            markeredgecolor='black', markeredgewidth=1, label='Start', zorder=5)
    ax.plot(x[-1], y_pos[-1], 's', color='red', markersize=8,
            markeredgecolor='black', markeredgewidth=1, label='End', zorder=5)

    # Axis limits: zoom into confinement region
    lim = R_w + zoom_padding
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Cartesian Trajectory ({system.confinement.upper()} confinement)\n'
        f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f},  '
        f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f},  '
        f'$\\tilde{{k}}$ = {system.k_tilde:.1f}',
        fontsize=14)

    plt.tight_layout()
    return fig


def _plot_cartesian_trajectories(system, initial_conditions,
                                  t_span=(0, 100),
                                  zoom_padding=1.5, wall_depth=2.0,
                                  figsize=(12, 12)):
    """
    Plot multiple trajectories on one Cartesian (x, y) plot with blue
    annular wall shading matching phase portrait styling.

    Parameters
    ----------
    initial_conditions : list of (r0, theta0, phi0) tuples
        Starting states for each trajectory.
    """
    fig, ax = plt.subplots(figsize=figsize)
    R_w = system.R_w_tilde

    # Wall shading
    n_ring = 200
    wall_theta = np.linspace(0, 2 * np.pi, n_ring)

    if system.confinement == 'inside':
        r_inner, r_outer = R_w, R_w + wall_depth
    else:
        r_inner, r_outer = max(0, R_w - wall_depth), R_w

    theta_fill = np.linspace(0, 2 * np.pi, n_ring)
    x_inner = r_inner * np.cos(theta_fill)
    y_inner = r_inner * np.sin(theta_fill)
    x_outer = r_outer * np.cos(theta_fill)
    y_outer = r_outer * np.sin(theta_fill)

    ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
            np.concatenate([y_outer, y_inner[::-1]]),
            color='blue', alpha=0.15, label='Wall region')
    ax.plot(R_w * np.cos(wall_theta), R_w * np.sin(wall_theta),
            color='blue', linewidth=2,
            label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.1f})')

    # Trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))

    for ic, col in zip(initial_conditions, colors):
        t, r, theta, phi = integrate_full_trajectory(system, ic, t_span)
        x = r * np.cos(phi)
        y_pos = r * np.sin(phi)

        # Trajectory line (zorder=3 keeps it above the blue wall shading)
        ax.plot(x, y_pos, '-', color=col, linewidth=0.6, alpha=0.8, zorder=3)

        # Direction arrows along trajectory
        n_arrows = 8
        arrow_indices = np.linspace(0, len(x) - 2, n_arrows, dtype=int)
        for idx in arrow_indices:
            dx = x[idx + 1] - x[idx]
            dy = y_pos[idx + 1] - y_pos[idx]
            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                ax.annotate('', xy=(x[idx + 1], y_pos[idx + 1]),
                            xytext=(x[idx], y_pos[idx]),
                            arrowprops=dict(arrowstyle='->', color=col,
                                            lw=1.2, mutation_scale=12),
                            zorder=4)

        ax.plot(x[0], y_pos[0], 'o', color=col, markersize=10,
                markeredgecolor='black', markeredgewidth=1,
                label=f'IC: ($r_0$={ic[0]:.2f}, $\\theta_0$={ic[1]:.2f})',
                zorder=5)
        ax.plot(x[-1], y_pos[-1], 's', color=col, markersize=7,
                markeredgecolor='black', markeredgewidth=1, zorder=5)

    # Axis limits: zoom into confinement region
    lim = R_w + zoom_padding
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Cartesian Trajectories ({system.confinement.upper()} confinement)\n'
        f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f},  '
        f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f},  '
        f'$\\tilde{{k}}$ = {system.k_tilde:.1f}',
        fontsize=14)

    plt.tight_layout()
    return fig


def _plot_polar_phase_portrait_streamlines(system, r_range=None,
                                            resolution=50, density=2.0,
                                            near_wall_zoom=False,
                                            show_fixed_points=True,
                                            figsize=(10, 10)):
    """
    Polar phase portrait using real matplotlib streamlines.

    Method: run streamplot on a hidden rectangular (r, theta) axis,
    reconstruct continuous paths, transform each path to polar
    coordinates (theta, r), and plot with velocity-coloured lines
    and direction arrows.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as mpl_cm

    R_w = system.R_w_tilde

    # Set r range
    if r_range is None:
        if near_wall_zoom:
            if system.confinement == 'inside':
                r_range = (max(0.01, R_w - 0.5), R_w + 1.5)
            else:
                r_range = (max(0.01, R_w - 1.5), R_w + 0.5)
        else:
            if system.confinement == 'inside':
                r_range = (max(0.01, R_w - 1), R_w + 2)
            else:
                r_range = (max(0.01, R_w - 2), R_w + 1)

    # Compute vector field on rectangular grid
    r = np.linspace(r_range[0], r_range[1], resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    R, THETA = np.meshgrid(r, theta)
    dR, dTHETA = system.vector_field(R, THETA)
    speed = np.sqrt(dR**2 + dTHETA**2)

    speed_min = np.min(speed)
    speed_max = np.max(speed)
    norm = Normalize(vmin=speed_min, vmax=speed_max)
    cmap = mpl_cm.viridis

    # Hidden rectangular streamplot to extract paths
    fig_hidden, ax_hidden = plt.subplots(figsize=(6, 6))
    ax_hidden.set_xlim(r_range[0], r_range[1])
    ax_hidden.set_ylim(0, 2*np.pi)
    strm = ax_hidden.streamplot(R, THETA, dR, dTHETA,
                                 color=speed, cmap='viridis',
                                 density=density, linewidth=0.8)

    # Extract segments and reconstruct continuous paths
    segments = strm.lines.get_segments()
    paths = []
    if len(segments) > 0:
        current_path = list(segments[0])
        for i in range(1, len(segments)):
            seg = segments[i]
            if np.allclose(current_path[-1], seg[0], atol=1e-10):
                current_path.append(seg[1])
            else:
                if len(current_path) >= 2:
                    paths.append(np.array(current_path))
                current_path = list(seg)
        if len(current_path) >= 2:
            paths.append(np.array(current_path))

    plt.close(fig_hidden)

    # Create polar figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Wall shading FIRST (low zorder so streamlines render on top)
    wall_theta_arr = np.linspace(0, 2*np.pi, 300)

    if system.confinement == 'inside':
        ax.fill_between(wall_theta_arr, R_w, r_range[1],
                         alpha=0.15, color='blue', zorder=1, label='Wall region')
    else:
        ax.fill_between(wall_theta_arr, r_range[0], R_w,
                         alpha=0.15, color='blue', zorder=1, label='Wall region')

    # Wall circle
    ax.plot(wall_theta_arr, np.full_like(wall_theta_arr, R_w), 'b-',
            linewidth=2.5, zorder=2,
            label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.2f})')

    # Plot each reconstructed streamline path on polar axis
    for path in paths:
        r_path = path[:, 0]
        theta_path = path[:, 1]

        # Compute speed at each point for colour mapping
        dr_vals = np.array([system.dr_dt(np.array([rr]), np.array([tt]))[0]
                           for rr, tt in zip(r_path, theta_path)])
        dth_vals = np.array([system.dtheta_dt(np.array([rr]), np.array([tt]))[0]
                            for rr, tt in zip(r_path, theta_path)])
        spd = np.sqrt(dr_vals**2 + dth_vals**2)

        # Create coloured line segments on polar axis (theta, r)
        points = np.column_stack([theta_path, r_path])
        segs = np.array([points[i:i+2] for i in range(len(points)-1)])

        if len(segs) > 0:
            lc = LineCollection(segs, cmap='viridis', norm=norm,
                               linewidth=0.8, zorder=3)
            lc.set_array(spd[:-1])
            ax.add_collection(lc)

        # Direction arrow at ~30% along the path
        if len(r_path) > 3:
            arrow_idx = max(1, len(r_path) // 3)
            t1, r1 = theta_path[arrow_idx-1], r_path[arrow_idx-1]
            t2, r2 = theta_path[arrow_idx], r_path[arrow_idx]

            arrow_spd = spd[arrow_idx]
            arrow_color = cmap(norm(arrow_spd))

            ax.annotate('', xy=(t2, r2), xytext=(t1, r1),
                        arrowprops=dict(arrowstyle='->', color=arrow_color,
                                        lw=1.2, mutation_scale=12),
                        zorder=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('|velocity|', fontsize=11)

    # Fixed points
    if show_fixed_points:
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)

        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                marker, color = 'o', 'limegreen'
            elif 'Saddle' in result['classification']:
                marker, color = 'x', 'red'
            else:
                marker, color = 's', 'orange'

            ax.plot(theta_fp, r_fp, marker, markersize=10,
                    markeredgewidth=2, color=color, markeredgecolor='black',
                    alpha=0.6,
                    label=f"{result['classification']} ({r_fp:.2f}, {theta_fp:.2f})",
                    zorder=6)

    # Configure polar axis
    ax.set_rlim(r_range[0], r_range[1])
    ax.set_title(
        f'Polar Phase Portrait ({system.confinement.upper()} confinement)\n'
        f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f},  '
        f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.2f},  '
        f'$\\tilde{{k}}$ = {system.k_tilde:.1f}',
        fontsize=13, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.1), fontsize=9)

    plt.tight_layout()
    return fig

if __name__ == "__main__":

    # =========================================================================
    # CREATE OUTPUT FOLDER
    # =========================================================================
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}/")
    else:
        print(f"Using existing folder: {OUTPUT_FOLDER}/")

    # =========================================================================
    # PRINT CONFIGURATION
    # =========================================================================
    print("="*70)
    print("dcABP Circular Wall Analysis Playground")
    print("="*70)
    print(f"\nSystem Parameters:")
    print(f"  omega~      = {OMEGA_TILDE}")
    print(f"  R~_w        = {R_W_TILDE}")
    print(f"  k~          = {K_TILDE}")
    print(f"  Confinement = {CONFINEMENT}")
    print(f"  r~ range    = {R_RANGE}")
    print(f"  Resolution  = {RESOLUTION}")
    print(f"  Density     = {DENSITY}")
    print(f"\nOutput folder: {OUTPUT_FOLDER}/")

    print(f"\nEnabled Features:")
    print(f"  Stability Analysis        : {'Y' if RUN_STABILITY_ANALYSIS else 'N'}")
    print(f"  Cartesian Phase Portrait  : {'Y' if RUN_PHASE_PORTRAIT else 'N'}")
    print(f"    - Show Nullclines       : {'Y' if SHOW_NULLCLINES else 'N'}")
    print(f"    - Show Fixed Points     : {'Y' if SHOW_FIXED_POINTS else 'N'}")
    print(f"  Polar Streamlines         : {'Y' if RUN_POLAR_STREAMLINES else 'N'}")
    print(f"  Polar Quiver              : {'Y' if RUN_POLAR_QUIVER else 'N'}")
    print(f"  Cartesian vs Polar        : {'Y' if RUN_CARTESIAN_VS_POLAR else 'N'}")
    print(f"  omega~ Sweep              : {'Y' if RUN_OMEGA_SWEEP else 'N'}")
    print(f"  R~_w Sweep                : {'Y' if RUN_R_W_SWEEP else 'N'}")
    print(f"  k~ Sweep                  : {'Y' if RUN_K_SWEEP else 'N'}")
    print(f"  Bifurcation (omega~)      : {'Y' if RUN_BIFURCATION_OMEGA else 'N'}")
    print(f"  Bifurcation (R~_w)        : {'Y' if RUN_BIFURCATION_R_W else 'N'}")
    print(f"  Trajectories (phase)      : {'Y' if RUN_TRAJECTORIES else 'N'}")
    print(f"  Cartesian Trajectory (1)  : {'Y' if RUN_CARTESIAN_TRAJECTORY else 'N'}")
    print(f"  Cartesian Trajectories (N): {'Y' if RUN_CARTESIAN_TRAJECTORIES else 'N'}")
    print(f"  Inside/Outside Comparison : {'Y' if RUN_INSIDE_OUTSIDE_COMPARISON else 'N'}")
    print(f"  --- Supervisor Analysis ---")
    print(f"  Basin of Attraction       : {'Y' if RUN_BASIN_OF_ATTRACTION else 'N'}")
    print(f"  sin(theta*) Analysis      : {'Y' if RUN_SIN_THETA_ANALYSIS else 'N'}")
    print(f"  Two-Parameter Phase Diag  : {'Y' if RUN_PHASE_DIAGRAM else 'N'}")

    # =========================================================================
    # CREATE MAIN SYSTEM
    # =========================================================================
    system = dcABPCircularWall(OMEGA_TILDE, R_W_TILDE, K_TILDE, CONFINEMENT)

    # =========================================================================
    # STABILITY ANALYSIS
    # =========================================================================
    if RUN_STABILITY_ANALYSIS:
        print("\n" + "="*70)
        print("STABILITY ANALYSIS")
        print("="*70)
        results = full_stability_analysis(system, verbose=True)

    # =========================================================================
    # CARTESIAN PHASE PORTRAIT (uses density-aware override)
    # =========================================================================
    if RUN_PHASE_PORTRAIT:
        print("\n" + "="*70)
        print("CARTESIAN PHASE PORTRAIT")
        print("="*70)
        print("Generating main phase portrait...")
        fig = _plot_phase_portrait_with_density(system, r_range=R_RANGE,
                                                theta_range=THETA_RANGE,
                                                resolution=RESOLUTION, density=DENSITY,
                                                figsize=MAIN_FIGSIZE,
                                                show_nullclines=SHOW_NULLCLINES,
                                                show_fixed_points=SHOW_FIXED_POINTS)
        filepath = os.path.join(OUTPUT_FOLDER, 'phase_portrait_cartesian.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # POLAR PHASE PORTRAITS (real streamlines on polar axis)
    # =========================================================================
    if RUN_POLAR_STREAMLINES:
        print("\n" + "="*70)
        print("POLAR PHASE PORTRAIT (streamlines)")
        print("="*70)
        if POLAR_NEAR_WALL_ZOOM:
            print("Note: Near-wall zoom ENABLED")
        print("Generating polar phase portrait with real streamlines...")
        fig = _plot_polar_phase_portrait_streamlines(
            system, r_range=R_RANGE,
            resolution=RESOLUTION, density=DENSITY,
            near_wall_zoom=POLAR_NEAR_WALL_ZOOM,
            show_fixed_points=SHOW_FIXED_POINTS,
            figsize=POLAR_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'phase_portrait_polar_streamlines.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_POLAR_QUIVER:
        print("\nGenerating polar phase portrait (quiver)...")
        fig = plot_polar_phase_portrait_quiver(system, resolution=POLAR_QUIVER_RESOLUTION,
                                                figsize=POLAR_FIGSIZE,
                                                near_wall_zoom=POLAR_NEAR_WALL_ZOOM)
        filepath = os.path.join(OUTPUT_FOLDER, 'phase_portrait_polar_quiver.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_CARTESIAN_VS_POLAR:
        print("\nGenerating Cartesian vs Polar comparison...")
        fig = compare_cartesian_vs_polar_phase_portrait(system,
                                                         figsize=CARTESIAN_VS_POLAR_FIGSIZE,
                                                         near_wall_zoom=POLAR_NEAR_WALL_ZOOM)
        filepath = os.path.join(OUTPUT_FOLDER, 'comparison_cartesian_vs_polar.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # PARAMETER SWEEPS
    # =========================================================================
    if RUN_OMEGA_SWEEP:
        print("\n" + "="*70)
        print("PARAMETER SWEEP: omega~")
        print("="*70)
        fig = parameter_sweep_omega(R_w_tilde=R_W_TILDE, k_tilde=K_TILDE,
                                     omega_values=OMEGA_SWEEP_VALUES,
                                     confinement=CONFINEMENT, figsize=PANEL_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sweep_omega.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_R_W_SWEEP:
        print("\n" + "="*70)
        print("PARAMETER SWEEP: R~_w")
        print("="*70)
        fig = parameter_sweep_R_w(omega_tilde=OMEGA_TILDE, k_tilde=K_TILDE,
                                   R_w_values=R_W_SWEEP_VALUES,
                                   confinement=CONFINEMENT, figsize=PANEL_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sweep_Rw.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_K_SWEEP:
        print("\n" + "="*70)
        print("PARAMETER SWEEP: k~")
        print("="*70)
        fig = parameter_sweep_k(omega_tilde=OMEGA_TILDE, R_w_tilde=R_W_TILDE,
                                 k_values=K_SWEEP_VALUES,
                                 confinement=CONFINEMENT, figsize=PANEL_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sweep_k.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # BIFURCATION DIAGRAMS
    # =========================================================================
    if RUN_BIFURCATION_OMEGA:
        print("\n" + "="*70)
        print("BIFURCATION DIAGRAM: vs omega~")
        print("="*70)
        fig = bifurcation_diagram_omega(R_w_tilde=R_W_TILDE, k_tilde=K_TILDE,
                                         omega_range=OMEGA_BIFURCATION_RANGE,
                                         n_points=BIFURCATION_POINTS,
                                         confinement=CONFINEMENT)
        filepath = os.path.join(OUTPUT_FOLDER, 'bifurcation_omega.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_BIFURCATION_R_W:
        print("\n" + "="*70)
        print("BIFURCATION DIAGRAM: vs R~_w")
        print("="*70)
        fig = bifurcation_diagram_R_w(omega_tilde=OMEGA_TILDE, k_tilde=K_TILDE,
                                       R_w_range=R_W_BIFURCATION_RANGE,
                                       n_points=BIFURCATION_POINTS,
                                       confinement=CONFINEMENT)
        filepath = os.path.join(OUTPUT_FOLDER, 'bifurcation_Rw.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # TRAJECTORIES ON PHASE PORTRAIT (uses density-aware override)
    # =========================================================================
    if RUN_TRAJECTORIES:
        print("\n" + "="*70)
        print("TRAJECTORY INTEGRATION (phase space)")
        print("="*70)
        print(f"ICs: {TRAJECTORY_INITIAL_CONDITIONS}")
        fig = _plot_trajectories_on_phase_portrait_with_density(
            system, TRAJECTORY_INITIAL_CONDITIONS,
            r_range=R_RANGE, t_span=TRAJECTORY_TIME_SPAN,
            resolution=RESOLUTION, density=DENSITY,
            figsize=MAIN_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'trajectories.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # SINGLE CARTESIAN TRAJECTORY (improved styling)
    # =========================================================================
    if RUN_CARTESIAN_TRAJECTORY:
        print("\n" + "="*70)
        print("CARTESIAN TRAJECTORY (single)")
        print("="*70)
        print(f"IC: {CARTESIAN_TRAJECTORY_IC}")
        fig = _plot_cartesian_trajectory_improved(
            system, CARTESIAN_TRAJECTORY_IC,
            t_span=CARTESIAN_TRAJECTORY_TIME,
            zoom_padding=CARTESIAN_ZOOM_PADDING,
            wall_depth=CARTESIAN_WALL_DEPTH,
            figsize=CARTESIAN_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'trajectory_cartesian_xy.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # MULTI CARTESIAN TRAJECTORIES (all ICs on one x,y plot)
    # =========================================================================
    if RUN_CARTESIAN_TRAJECTORIES:
        print("\n" + "="*70)
        print("CARTESIAN TRAJECTORIES (multi)")
        print("="*70)

        # Convert (r0, theta0) pairs to (r0, theta0, phi0=0) triples
        CARTESIAN_ICS = [(r0, th0, 0.0) for r0, th0 in TRAJECTORY_INITIAL_CONDITIONS]

        print(f"Generating Cartesian trajectories with ICs:")
        for ic in CARTESIAN_ICS:
            print(f"  (r0={ic[0]:.2f}, theta0={ic[1]:.2f}, phi0={ic[2]:.2f})")

        fig = _plot_cartesian_trajectories(
            system, CARTESIAN_ICS,
            t_span=TRAJECTORY_TIME_SPAN,
            zoom_padding=CARTESIAN_ZOOM_PADDING,
            wall_depth=CARTESIAN_WALL_DEPTH,
            figsize=CARTESIAN_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'cartesian_trajectories_multi.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # INSIDE VS OUTSIDE COMPARISON
    # =========================================================================
    if RUN_INSIDE_OUTSIDE_COMPARISON:
        print("\n" + "="*70)
        print("INSIDE vs OUTSIDE CONFINEMENT COMPARISON")
        print("="*70)
        fig = compare_inside_outside(omega_tilde=OMEGA_TILDE, R_w_tilde=R_W_TILDE,
                                      k_tilde=K_TILDE, figsize=COMPARISON_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'comparison_inside_outside.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # BASIN OF ATTRACTION (Supervisor Question 1)
    # =========================================================================
    if RUN_BASIN_OF_ATTRACTION:
        print("\n" + "="*70)
        print("BASIN OF ATTRACTION ANALYSIS")
        print("="*70)
        basin_data = compute_basin_of_attraction(system, r_range=R_RANGE,
                                                  theta_range=THETA_RANGE,
                                                  n_r=BASIN_N_R, n_theta=BASIN_N_THETA,
                                                  t_max=BASIN_T_MAX, verbose=True)
        fig = plot_basin_of_attraction(basin_data, figsize=BASIN_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'basin_of_attraction.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # sin(theta*) ANALYSIS (Supervisor Question 2)
    # =========================================================================
    if RUN_SIN_THETA_ANALYSIS:
        print("\n" + "="*70)
        print("sin(theta*) SYSTEMATIC ANALYSIS")
        print("="*70)
        print("Computing across parameter space...")
        sin_data = analyze_sin_theta_branches(R_w_range=SIN_ANALYSIS_RW_RANGE,
                                               omega_range=SIN_ANALYSIS_OMEGA_RANGE,
                                               n_Rw=SIN_ANALYSIS_N_RW,
                                               n_omega=SIN_ANALYSIS_N_OMEGA,
                                               confinement=CONFINEMENT,
                                               k_tilde=K_TILDE)
        fig = plot_sin_theta_analysis(sin_data, figsize=SIN_ANALYSIS_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sin_theta_analysis.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # TWO-PARAMETER PHASE DIAGRAM (Supervisor Question 3)
    # =========================================================================
    if RUN_PHASE_DIAGRAM:
        print("\n" + "="*70)
        print("TWO-PARAMETER PHASE DIAGRAM")
        print("="*70)
        print("Computing in (omega~, 1/R~_w) space...")
        phase_data = compute_phase_diagram(omega_range=PHASE_DIAGRAM_OMEGA_RANGE,
                                            inv_Rw_range=PHASE_DIAGRAM_INV_RW_RANGE,
                                            n_omega=PHASE_DIAGRAM_N_OMEGA,
                                            n_inv_Rw=PHASE_DIAGRAM_N_INV_RW,
                                            confinement=CONFINEMENT,
                                            k_tilde=K_TILDE)
        fig = plot_phase_diagram(phase_data, figsize=PHASE_DIAGRAM_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'phase_diagram_2param.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll figures saved to: {OUTPUT_FOLDER}/")
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith('.png'):
            print(f"  - {f}")
