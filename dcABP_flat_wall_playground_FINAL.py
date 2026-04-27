"""
Doubly Chiral Active Brownian Particles (dcABP) - Flat Wall Playground (Complete)
==================================================================================

A user-friendly analysis tool for dcABPs near a flat wall.

HOW TO USE:
-----------
1. Set the OUTPUT_FOLDER name for your run
2. Adjust the SYSTEM PARAMETERS (omega~, k~)
3. Set Y/N toggles (True/False) for which analyses you want
4. Adjust any specific settings for enabled analyses
5. Run: python dcABP_flat_wall_playground_FINAL.py

All adjustable parameters are at the top of the file for easy modification.
Results are automatically saved to your custom-named folder.

FEATURES AVAILABLE:
-------------------
- Core: Stability analysis, phase portraits, nullclines
- Parameter sweeps: omega~ sweep, k~ sweep
- Bifurcation diagrams
- Trajectory integration and time evolution
- Basin of attraction analysis (supervisor question 1)
- sin(theta*) systematic analysis (supervisor question 2)
- Two-parameter phase diagram (supervisor question 3)

Author: Tungsten (Masters Project)
Supervisor: Jaime Agudo-Canalejo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from dcABP_flat_wall_analysis_complete import *
except ImportError:
    print("ERROR: Could not import dcABP_flat_wall_analysis_complete")
    print("Make sure the master file is in the same directory.")
    sys.exit(1)


OUTPUT_FOLDER = "limiting_case3"   # Change this for each new run!

# -----------------------------------------------------------------------------
# MAIN SYSTEM PARAMETERS
# -----------------------------------------------------------------------------
OMEGA_TILDE = -0.0         # Rescaled intrinsic frequency: omega~ = omega/(alpha*v)
                            # For fixed points: need -1 < omega~ < 0
K_TILDE = 1.0               # Rescaled wall stiffness: k~ = mu*k/(alpha*v)

# -----------------------------------------------------------------------------
# PHASE PORTRAIT SETTINGS
# -----------------------------------------------------------------------------
X_RANGE = (-1.5, 0.5)       # Range of x~ values (wall at x=0)
THETA_RANGE = (0, 2*np.pi)  # Range of theta values
RESOLUTION = 30             # Grid resolution for streamplot

RUN_STABILITY_ANALYSIS = True       # Print stability analysis to console
RUN_PHASE_PORTRAIT = True           # Main (x~, theta) phase portrait
SHOW_NULLCLINES = False              # Show nullclines on phase portrait
SHOW_FIXED_POINTS = True            # Show fixed points on phase portrait

# --- Parameter Sweeps ---
RUN_OMEGA_SWEEP = False             # Sweep over omega~ values
RUN_K_SWEEP = False                 # Sweep over k~ values

# --- Bifurcation Diagram ---
RUN_BIFURCATION = False             # Bifurcation diagram: theta* and x* vs omega~

# --- Trajectory Analysis ---
RUN_TRAJECTORIES = True            # Plot trajectories on phase portrait
RUN_TIME_EVOLUTION = True          # Plot x(t) and theta(t) vs time

# --- SUPERVISOR ANALYSIS (Advanced) ---
RUN_BASIN_OF_ATTRACTION = False     # Basin of attraction analysis
RUN_SIN_THETA_ANALYSIS = False      # Systematic sin(theta*) analysis
RUN_PHASE_DIAGRAM = False           # Two-parameter phase diagram

OMEGA_SWEEP_VALUES = [-0.9, -0.5, -0.2, -0.05]
K_SWEEP_VALUES = [0.5, 1.0, 2.0, 5.0]

# --- Bifurcation Diagram Settings ---
OMEGA_BIFURCATION_RANGE = (-0.99, -0.01)
BIFURCATION_POINTS = 100

# --- Trajectory Settings ---
TRAJECTORY_INITIAL_CONDITIONS = [
    (-0.5, np.pi/2),      # (x0, theta0)
    (-0.5, np.pi),
    (-0.5, 3*np.pi/2),
    (0.3, np.pi),
]
TRAJECTORY_TIME_SPAN = (0, 50)
TIME_EVOLUTION_START = (0.3, np.pi)
TIME_EVOLUTION_DURATION = 50

# --- Basin of Attraction Settings ---
BASIN_N_X = 30              # Number of x points in grid
BASIN_N_THETA = 30          # Number of theta points in grid
BASIN_T_MAX = 200           # Maximum integration time

# --- sin(theta*) Analysis Settings ---
SIN_ANALYSIS_N_OMEGA = 50   # Grid resolution in omega~
SIN_ANALYSIS_N_K = 50       # Grid resolution in k~
SIN_ANALYSIS_OMEGA_RANGE = (-0.99, -0.01)
SIN_ANALYSIS_K_RANGE = (0.1, 5.0)

# --- Phase Diagram Settings ---
PHASE_DIAGRAM_N_OMEGA = 100
PHASE_DIAGRAM_N_K = 100
PHASE_DIAGRAM_OMEGA_RANGE = (-1.5, 0.5)
PHASE_DIAGRAM_K_RANGE = (0.1, 5.0)

# --- Figure Settings ---
FIGURE_DPI = 150
MAIN_FIGSIZE = (10, 8)
PANEL_FIGSIZE = (15, 10)
BIFURCATION_FIGSIZE = (12, 5)
BASIN_FIGSIZE = (14, 6)
SIN_ANALYSIS_FIGSIZE = (14, 10)
PHASE_DIAGRAM_FIGSIZE = (14, 10)


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
    print("dcABP Flat Wall Analysis Playground")
    print("="*70)
    print(f"\nSystem Parameters:")
    print(f"  omega~  = {OMEGA_TILDE}")
    print(f"  k~      = {K_TILDE}")
    print(f"  x~ range = {X_RANGE}")
    print(f"\nOutput folder: {OUTPUT_FOLDER}/")

    print(f"\nEnabled Features:")
    print(f"  Stability Analysis      : {'Y' if RUN_STABILITY_ANALYSIS else 'N'}")
    print(f"  Phase Portrait          : {'Y' if RUN_PHASE_PORTRAIT else 'N'}")
    print(f"    - Show Nullclines     : {'Y' if SHOW_NULLCLINES else 'N'}")
    print(f"    - Show Fixed Points   : {'Y' if SHOW_FIXED_POINTS else 'N'}")
    print(f"  omega~ Sweep            : {'Y' if RUN_OMEGA_SWEEP else 'N'}")
    print(f"  k~ Sweep                : {'Y' if RUN_K_SWEEP else 'N'}")
    print(f"  Bifurcation Diagram     : {'Y' if RUN_BIFURCATION else 'N'}")
    print(f"  Trajectories            : {'Y' if RUN_TRAJECTORIES else 'N'}")
    print(f"  Time Evolution          : {'Y' if RUN_TIME_EVOLUTION else 'N'}")
    print(f"  --- Supervisor Analysis ---")
    print(f"  Basin of Attraction     : {'Y' if RUN_BASIN_OF_ATTRACTION else 'N'}")
    print(f"  sin(theta*) Analysis    : {'Y' if RUN_SIN_THETA_ANALYSIS else 'N'}")
    print(f"  Two-Parameter Phase Diag: {'Y' if RUN_PHASE_DIAGRAM else 'N'}")

    # =========================================================================
    # CREATE MAIN SYSTEM
    # =========================================================================
    system = dcABPFlatWall(OMEGA_TILDE, K_TILDE)

    # =========================================================================
    # STABILITY ANALYSIS
    # =========================================================================
    if RUN_STABILITY_ANALYSIS:
        print("\n" + "="*70)
        print("STABILITY ANALYSIS")
        print("="*70)
        results = stability_analysis(system, verbose=True)

    # =========================================================================
    # PHASE PORTRAIT
    # =========================================================================
    if RUN_PHASE_PORTRAIT:
        print("\n" + "="*70)
        print("PHASE PORTRAIT")
        print("="*70)
        print("Generating main phase portrait...")
        fig = plot_phase_portrait(system, x_range=X_RANGE, theta_range=THETA_RANGE,
                                   resolution=RESOLUTION, figsize=MAIN_FIGSIZE,
                                   show_nullclines=SHOW_NULLCLINES,
                                   show_fixed_points=SHOW_FIXED_POINTS)
        filepath = os.path.join(OUTPUT_FOLDER, 'phase_portrait.png')
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
        print(f"Values: {OMEGA_SWEEP_VALUES}")
        fig = parameter_sweep_omega(k_tilde=K_TILDE, omega_values=OMEGA_SWEEP_VALUES,
                                     figsize=PANEL_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sweep_omega.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    if RUN_K_SWEEP:
        print("\n" + "="*70)
        print("PARAMETER SWEEP: k~")
        print("="*70)
        print(f"Values: {K_SWEEP_VALUES}")
        fig = parameter_sweep_k(omega_tilde=OMEGA_TILDE, k_values=K_SWEEP_VALUES,
                                 figsize=PANEL_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'sweep_k.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # BIFURCATION DIAGRAM
    # =========================================================================
    if RUN_BIFURCATION:
        print("\n" + "="*70)
        print("BIFURCATION DIAGRAM")
        print("="*70)
        fig = bifurcation_diagram(k_tilde=K_TILDE, omega_range=OMEGA_BIFURCATION_RANGE,
                                   n_points=BIFURCATION_POINTS, figsize=BIFURCATION_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'bifurcation_diagram.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # TRAJECTORIES
    # =========================================================================
    if RUN_TRAJECTORIES:
        print("\n" + "="*70)
        print("TRAJECTORY INTEGRATION")
        print("="*70)
        print(f"Initial conditions: {TRAJECTORY_INITIAL_CONDITIONS}")
        fig = plot_trajectories_on_phase_portrait(system, TRAJECTORY_INITIAL_CONDITIONS,
                                                   x_range=X_RANGE, t_span=TRAJECTORY_TIME_SPAN,
                                                   figsize=MAIN_FIGSIZE)
        filepath = os.path.join(OUTPUT_FOLDER, 'trajectories.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    # =========================================================================
    # TIME EVOLUTION
    # =========================================================================
    if RUN_TIME_EVOLUTION:
        print("\n" + "="*70)
        print("TIME EVOLUTION")
        print("="*70)
        print(f"Starting from: {TIME_EVOLUTION_START}")

        t, x, theta = integrate_trajectory(system, TIME_EVOLUTION_START,
                                            (0, TIME_EVOLUTION_DURATION))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(t, x, 'b-', linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle='--', label='Wall')
        ax1.set_ylabel(r'$\tilde{x}(t)$')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, theta, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel(r'$\theta(t)$')
        ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax2.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Time Evolution from IC = {TIME_EVOLUTION_START}')
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_FOLDER, 'time_evolution.png')
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
        basin_data = compute_basin_of_attraction(system, x_range=X_RANGE,
                                                  theta_range=THETA_RANGE,
                                                  n_x=BASIN_N_X, n_theta=BASIN_N_THETA,
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
        sin_data = analyze_sin_theta_flat_wall(omega_range=SIN_ANALYSIS_OMEGA_RANGE,
                                                k_range=SIN_ANALYSIS_K_RANGE,
                                                n_omega=SIN_ANALYSIS_N_OMEGA,
                                                n_k=SIN_ANALYSIS_N_K)
        fig = plot_sin_theta_analysis_flat_wall(sin_data, figsize=SIN_ANALYSIS_FIGSIZE)
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
        print("Computing in (omega~, k~) space...")
        phase_data = compute_phase_diagram_flat_wall(omega_range=PHASE_DIAGRAM_OMEGA_RANGE,
                                                      k_range=PHASE_DIAGRAM_K_RANGE,
                                                      n_omega=PHASE_DIAGRAM_N_OMEGA,
                                                      n_k=PHASE_DIAGRAM_N_K)
        fig = plot_phase_diagram_flat_wall(phase_data, figsize=PHASE_DIAGRAM_FIGSIZE)
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
