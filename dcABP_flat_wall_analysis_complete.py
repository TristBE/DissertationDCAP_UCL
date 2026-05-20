
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Tuple, Optional, List, Dict
import warnings


# =============================================================================
# SECTION 1: SYSTEM DEFINITION
# =============================================================================

class dcABPFlatWall:
    
    def __init__(self, omega_tilde: float = -0.5, k_tilde: float = 1.0):
        self.omega_tilde = omega_tilde
        self.k_tilde = k_tilde
        
    def wall_force(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        F = np.zeros_like(x)
        mask = x < 0
        F[mask] = -self.k_tilde * x[mask]
        return F
    
    def wall_force_derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dF = np.zeros_like(x)
        dF[x < 0] = -self.k_tilde
        return dF
    
    def dx_dt(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return np.cos(theta) + self.wall_force(x)
    
    def dtheta_dt(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return 1 + self.omega_tilde + self.wall_force(x) * np.cos(theta)
    
    def vector_field(self, x: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.dx_dt(x, theta), self.dtheta_dt(x, theta)


# =============================================================================
# SECTION 2: FIXED POINT ANALYSIS
# =============================================================================

def find_fixed_points_detailed(system: dcABPFlatWall, 
                                verbose: bool = False) -> List[Dict]:
    omega = system.omega_tilde
    k = system.k_tilde
    
    results = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DETAILED FIXED POINT ANALYSIS - FLAT WALL")
        print(f"{'='*60}")
        print(f"Parameters: omega~ = {omega:.4f}, k~ = {k:.4f}")
    
    # Check existence condition: -1 < omega~ < 0
    if omega >= 0 or omega <= -1:
        if verbose:
            print(f"\nNo fixed points exist for omega~ = {omega:.3f}")
            print("  Require: -1 < omega~ < 0")
        return results
    
    # Calculate sin(theta*) values
    sqrt_neg_omega = np.sqrt(-omega)
    s_plus = sqrt_neg_omega      # Positive sin
    s_minus = -sqrt_neg_omega    # Negative sin
    
    if verbose:
        print(f"\nBranch solutions (sin(theta*) = +/- sqrt(-omega~)):")
        print(f"  s_plus  = +sqrt(-omega~) = {s_plus:.6f}")
        print(f"  s_minus = -sqrt(-omega~) = {s_minus:.6f}")
    
    # Process each branch
    for branch_name, s_value in [('plus', s_plus), ('minus', s_minus)]:
        
        result = {
            'branch': branch_name,
            'branch_value': s_value,
            'sin_theta_star': None,
            'cos_theta_star': None,
            'theta_star': None,
            'x_star': None,
            'valid': False,
            'validity_reason': None
        }
        
        
        
        theta_star = np.pi - np.arcsin(s_value)
        cos_theta_star = np.cos(theta_star)
        sin_theta_star = np.sin(theta_star)
        
        # Check cos(theta*) < 0 for wall contact
        if cos_theta_star >= 0:
            result['validity_reason'] = f'cos(theta*) = {cos_theta_star:.4f} >= 0 (no wall contact)'
            if verbose:
                print(f"\n  Branch '{branch_name}': INVALID - {result['validity_reason']}")
            results.append(result)
            continue
        
        x_star = cos_theta_star / k
        
        result['sin_theta_star'] = sin_theta_star
        result['cos_theta_star'] = cos_theta_star
        result['theta_star'] = theta_star
        result['x_star'] = x_star
        result['valid'] = True
        result['validity_reason'] = 'Valid fixed point'
        
        if verbose:
            print(f"\n  Branch '{branch_name}': VALID")
            print(f"    sin(theta*) = {sin_theta_star:.6f} ({'positive' if sin_theta_star > 0 else 'negative'})")
            print(f"    cos(theta*) = {cos_theta_star:.6f} ({'positive' if cos_theta_star > 0 else 'negative'})")
            print(f"    theta* = {theta_star:.4f} rad = {np.degrees(theta_star):.2f} deg")
            print(f"    x* = {x_star:.4f} (penetration into wall)")
        
        results.append(result)
    
    return results


def find_fixed_points(system: dcABPFlatWall) -> List[Tuple[float, float]]:
    detailed = find_fixed_points_detailed(system, verbose=False)
    return [(fp['x_star'], fp['theta_star']) for fp in detailed if fp['valid']]


# =============================================================================
# SECTION 3: JACOBIAN AND STABILITY ANALYSIS
# =============================================================================

def compute_jacobian(system: dcABPFlatWall, x: float, theta: float) -> np.ndarray:
    x_arr = np.array([x])
    F = system.wall_force(x_arr)[0]
    dF = system.wall_force_derivative(x_arr)[0]
    
    J = np.array([
        [dF, -np.sin(theta)],
        [dF * np.cos(theta), -F * np.sin(theta)]
    ])
    
    return J


def classify_fixed_point(jacobian: np.ndarray) -> Dict:
    eigenvalues = np.linalg.eigvals(jacobian)
    trace = np.trace(jacobian)
    det = np.linalg.det(jacobian)
    
    result = {
        'eigenvalues': eigenvalues,
        'trace': trace,
        'determinant': det,
        'classification': '',
        'stability': ''
    }
    
    # Classification following PHAS0049 Figure 8.1 logic
    if det < 0:
        result['classification'] = 'Saddle Point'
        result['stability'] = 'Unstable'
    elif det > 0:
        discriminant = trace**2 - 4*det
        if discriminant > 0:  # Real eigenvalues
            if trace < 0:
                result['classification'] = 'Stable Node'
                result['stability'] = 'Stable'
            elif trace > 0:
                result['classification'] = 'Unstable Node'
                result['stability'] = 'Unstable'
            else:
                result['classification'] = 'Non-isolated (tr=0, real)'
                result['stability'] = 'Marginal'
        elif discriminant < 0:  # Complex eigenvalues
            if trace < 0:
                result['classification'] = 'Stable Focus'
                result['stability'] = 'Stable'
            elif trace > 0:
                result['classification'] = 'Unstable Focus'
                result['stability'] = 'Unstable'
            else:
                result['classification'] = 'Centre'
                result['stability'] = 'Marginal'
        else:  # discriminant == 0, degenerate
            if trace < 0:
                result['classification'] = 'Stable Star/Improper Node'
                result['stability'] = 'Stable'
            elif trace > 0:
                result['classification'] = 'Unstable Star/Improper Node'
                result['stability'] = 'Unstable'
    else:  # det == 0
        result['classification'] = 'Non-simple (det=0)'
        result['stability'] = 'Marginal/Bifurcation'
    
    return result


def stability_analysis(system: dcABPFlatWall, verbose: bool = True) -> List[Dict]:
    fp_details = find_fixed_points_detailed(system, verbose=verbose)
    results = []
    
    if verbose:
        print(f"\n{'-'*60}")
        print("STABILITY CLASSIFICATION")
        print(f"{'-'*60}")
    
    for fp in fp_details:
        if not fp['valid']:
            results.append(fp)
            continue
        
        J = compute_jacobian(system, fp['x_star'], fp['theta_star'])
        stability = classify_fixed_point(J)
        
        fp.update(stability)
        fp['jacobian'] = J
        
        if verbose:
            print(f"\n  Branch '{fp['branch']}':")
            print(f"    Jacobian:")
            print(f"      [{J[0,0]:10.4f}  {J[0,1]:10.4f}]")
            print(f"      [{J[1,0]:10.4f}  {J[1,1]:10.4f}]")
            print(f"    Trace = {stability['trace']:.4f}")
            print(f"    Determinant = {stability['determinant']:.4f}")
            print(f"    Eigenvalues: {stability['eigenvalues']}")
            print(f"    Classification: {stability['classification']}")
            print(f"    Stability: {stability['stability']}")
        
        results.append(fp)
    
    return results


# =============================================================================
# SECTION 4: NULLCLINES
# =============================================================================

def compute_x_nullcline(system: dcABPFlatWall, theta_range: np.ndarray) -> np.ndarray:
    return -np.cos(theta_range) / system.k_tilde


def compute_theta_nullcline(system: dcABPFlatWall, x_range: np.ndarray) -> np.ndarray:
    theta_nullcline = np.full_like(x_range, np.nan)
    
    for i, x in enumerate(x_range):
        if x < 0:
            F = -system.k_tilde * x
            cos_theta = -(1 + system.omega_tilde) / F
            if -1 <= cos_theta <= 1:
                theta_nullcline[i] = np.arccos(cos_theta)
    
    return theta_nullcline


# =============================================================================
# SECTION 5: PHASE PORTRAIT PLOTTING
# =============================================================================

def plot_phase_portrait(system: dcABPFlatWall,
                        x_range: Tuple[float, float] = (-2, 1),
                        theta_range: Tuple[float, float] = (0, 2*np.pi),
                        resolution: int = 25,
                        figsize: Tuple[int, int] = (10, 8),
                        show_nullclines: bool = True,
                        show_fixed_points: bool = True,
                        title: Optional[str] = None) -> plt.Figure:
    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    X, THETA = np.meshgrid(x, theta)
    
    # Compute vector field
    dX, dTHETA = system.vector_field(X, THETA)
    speed = np.sqrt(dX**2 + dTHETA**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot streamlines
    strm = ax.streamplot(X, THETA, dX, dTHETA,
                         color=speed, cmap='viridis',
                         density=1.5, linewidth=0.8,
                         arrowsize=1.2)
    
    fig.colorbar(strm.lines, ax=ax, label='|velocity|')
    
    # Plot wall location
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-', label='Wall (x~=0)')
    ax.axvspan(x_range[0], 0, alpha=0.1, color='gray', label='Inside wall')
    
    # Plot nullclines
    if show_nullclines:
        theta_fine = np.linspace(theta_range[0], theta_range[1], 500)
        x_null = compute_x_nullcline(system, theta_fine)
        mask = (x_null >= x_range[0]) & (x_null <= 0)
        ax.plot(x_null[mask], theta_fine[mask], 'r-', linewidth=2, 
                label=r'$\dot{x}=0$ nullcline')
        
        # x >= 0 part: theta = pi/2, 3*pi/2
        if theta_range[0] <= np.pi/2 <= theta_range[1]:
            ax.axhline(y=np.pi/2, xmin=(0-x_range[0])/(x_range[1]-x_range[0]), 
                      xmax=1, color='r', linewidth=2, linestyle='--')
        if theta_range[0] <= 3*np.pi/2 <= theta_range[1]:
            ax.axhline(y=3*np.pi/2, xmin=(0-x_range[0])/(x_range[1]-x_range[0]),
                      xmax=1, color='r', linewidth=2, linestyle='--')
        
        # theta-nullcline
        x_fine = np.linspace(x_range[0], -0.01, 500)
        theta_null = compute_theta_nullcline(system, x_fine)
        valid = ~np.isnan(theta_null)
        if np.any(valid):
            ax.plot(x_fine[valid], theta_null[valid], 'b-', linewidth=2,
                    label=r'$\dot{\theta}=0$ nullcline')
            ax.plot(x_fine[valid], 2*np.pi - theta_null[valid], 'b-', linewidth=2)
    
    # Mark fixed points
    if show_fixed_points:
        results = stability_analysis(system, verbose=False)
        
        for fp in results:
            if not fp['valid']:
                continue
            
            if fp['stability'] == 'Stable':
                marker, color = 'o', 'green'
            elif 'Saddle' in fp['classification']:
                marker, color = 'x', 'red'
            else:
                marker, color = 's', 'orange'
            
            ax.plot(fp['x_star'], fp['theta_star'], marker, markersize=12, 
                    markeredgewidth=3, color=color,
                    label=f"{fp['classification']} ({fp['x_star']:.2f}, {fp['theta_star']:.2f})")
    
    # Labels and formatting
    ax.set_xlabel(r'$\tilde{x}$', fontsize=12)
    ax.set_ylabel(r'$\theta$', fontsize=12)
    ax.set_xlim(x_range)
    ax.set_ylim(theta_range)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    if title is None:
        title = f'dcABP Phase Portrait (Flat Wall)\n$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, $\\tilde{{k}}$ = {system.k_tilde:.2f}'
    ax.set_title(title, fontsize=12)
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6: PARAMETER SWEEPS
# =============================================================================

def parameter_sweep_omega(k_tilde: float = 1.0,
                          omega_values: Optional[List[float]] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if omega_values is None:
        omega_values = [-0.9, -0.5, -0.2, -0.05]
    
    n_plots = len(omega_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    x_range = (-1.5, 0.5)
    theta_range = (np.pi/4, 7*np.pi/4)
    
    for idx, omega in enumerate(omega_values):
        ax = axes[idx]
        system = dcABPFlatWall(omega_tilde=omega, k_tilde=k_tilde)
        
        x = np.linspace(x_range[0], x_range[1], 25)
        theta = np.linspace(theta_range[0], theta_range[1], 25)
        X, THETA = np.meshgrid(x, theta)
        dX, dTHETA = system.vector_field(X, THETA)
        speed = np.sqrt(dX**2 + dTHETA**2)
        
        ax.streamplot(X, THETA, dX, dTHETA, color=speed, cmap='viridis',
                     density=1.2, linewidth=0.6, arrowsize=1)
        
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.axvspan(x_range[0], 0, alpha=0.1, color='gray')
        
        # Fixed points
        results = stability_analysis(system, verbose=False)
        for fp in results:
            if fp['valid']:
                marker = 'o' if fp['stability'] == 'Stable' else 'x'
                color = 'green' if fp['stability'] == 'Stable' else 'red'
                ax.plot(fp['x_star'], fp['theta_star'], marker, markersize=10, 
                       markeredgewidth=2, color=color)
        
        ax.set_xlabel(r'$\tilde{x}$')
        ax.set_ylabel(r'$\theta$')
        ax.set_title(f'$\\tilde{{\\omega}}$ = {omega:.2f}')
        ax.set_yticks([np.pi/2, np.pi, 3*np.pi/2])
        ax.set_yticklabels([r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Parameter Sweep: $\\tilde{{\\omega}}$ (Flat Wall, $\\tilde{{k}}$ = {k_tilde})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def parameter_sweep_k(omega_tilde: float = -0.5,
                      k_values: Optional[List[float]] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if k_values is None:
        k_values = [0.5, 1.0, 2.0, 5.0]
    
    n_plots = len(k_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        system = dcABPFlatWall(omega_tilde=omega_tilde, k_tilde=k)
        
        # Adjust x_range based on k (penetration depth ~ 1/k)
        x_min = min(-2.0, -3.0/k)
        x_range = (x_min, 0.5)
        theta_range = (np.pi/4, 7*np.pi/4)
        
        x = np.linspace(x_range[0], x_range[1], 25)
        theta = np.linspace(theta_range[0], theta_range[1], 25)
        X, THETA = np.meshgrid(x, theta)
        dX, dTHETA = system.vector_field(X, THETA)
        speed = np.sqrt(dX**2 + dTHETA**2)
        
        ax.streamplot(X, THETA, dX, dTHETA, color=speed, cmap='viridis',
                     density=1.2, linewidth=0.6, arrowsize=1)
        
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.axvspan(x_range[0], 0, alpha=0.1, color='gray')
        
        results = stability_analysis(system, verbose=False)
        for fp in results:
            if fp['valid']:
                marker = 'o' if fp['stability'] == 'Stable' else 'x'
                color = 'green' if fp['stability'] == 'Stable' else 'red'
                ax.plot(fp['x_star'], fp['theta_star'], marker, markersize=10, 
                       markeredgewidth=2, color=color)
        
        ax.set_xlabel(r'$\tilde{x}$')
        ax.set_ylabel(r'$\theta$')
        ax.set_title(f'$\\tilde{{k}}$ = {k:.2f}')
        ax.set_yticks([np.pi/2, np.pi, 3*np.pi/2])
        ax.set_yticklabels([r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Parameter Sweep: $\\tilde{{k}}$ (Flat Wall, $\\tilde{{\\omega}}$ = {omega_tilde})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7: BIFURCATION DIAGRAMS
# =============================================================================

def bifurcation_diagram(k_tilde: float = 1.0,
                        omega_range: Tuple[float, float] = (-0.99, -0.01),
                        n_points: int = 100,
                        figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    omega_values = np.linspace(omega_range[0], omega_range[1], n_points)
    
    theta_stable = []
    theta_saddle = []
    x_stable = []
    x_saddle = []
    omega_stable = []
    omega_saddle = []
    
    for omega in omega_values:
        system = dcABPFlatWall(omega_tilde=omega, k_tilde=k_tilde)
        results = stability_analysis(system, verbose=False)
        
        for fp in results:
            if fp['valid']:
                if fp['stability'] == 'Stable':
                    omega_stable.append(omega)
                    x_stable.append(fp['x_star'])
                    theta_stable.append(fp['theta_star'])
                else:
                    omega_saddle.append(omega)
                    x_saddle.append(fp['x_star'])
                    theta_saddle.append(fp['theta_star'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # theta* vs omega~
    ax1.plot(omega_stable, theta_stable, 'g-', linewidth=2, label='Stable')
    ax1.plot(omega_saddle, theta_saddle, 'r--', linewidth=2, label='Saddle')
    ax1.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax1.set_ylabel(r'$\theta^*$', fontsize=12)
    ax1.set_title(r'Fixed Point Angle $\theta^*$ vs $\tilde{\omega}$')
    ax1.axhline(y=np.pi, color='gray', linestyle=':', alpha=0.5)
    ax1.set_yticks([np.pi/2, np.pi, 3*np.pi/2])
    ax1.set_yticklabels([r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # x* vs omega~
    ax2.plot(omega_stable, x_stable, 'g-', linewidth=2, label='Stable')
    ax2.plot(omega_saddle, x_saddle, 'r--', linewidth=2, label='Saddle')
    ax2.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax2.set_ylabel(r'$\tilde{x}^*$', fontsize=12)
    ax2.set_title(r'Fixed Point Position $\tilde{x}^*$ vs $\tilde{\omega}$')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Wall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Bifurcation Diagram (Flat Wall, $\\tilde{{k}}$ = {k_tilde})', fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 8: TRAJECTORY INTEGRATION
# =============================================================================

def integrate_trajectory(system: dcABPFlatWall,
                        initial_condition: Tuple[float, float],
                        t_span: Tuple[float, float] = (0, 50),
                        n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    def rhs(t, y):
        x, theta = y
        dx = system.dx_dt(np.array([x]), np.array([theta]))[0]
        dtheta = system.dtheta_dt(np.array([x]), np.array([theta]))[0]
        return [dx, dtheta]
    
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(rhs, t_span, initial_condition, t_eval=t_eval, method='RK45')
    
    return sol.t, sol.y[0], sol.y[1]


def plot_trajectories_on_phase_portrait(system: dcABPFlatWall,
                                         initial_conditions: List[Tuple[float, float]],
                                         x_range: Tuple[float, float] = (-1.5, 0.5),
                                         t_span: Tuple[float, float] = (0, 50),
                                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    
    fig = plot_phase_portrait(system, x_range=x_range, figsize=figsize, 
                              show_nullclines=False)
    ax = fig.axes[0]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))
    
    for ic, color in zip(initial_conditions, colors):
        t, x, theta = integrate_trajectory(system, ic, t_span)
        ax.plot(x, theta, '-', color=color, linewidth=2, alpha=0.8)
        ax.plot(ic[0], ic[1], 'o', color=color, markersize=10, 
                label=f'IC: ({ic[0]:.1f}, {ic[1]:.2f})')
        ax.plot(x[-1], theta[-1], 's', color=color, markersize=8)
    
    ax.legend(loc='upper right', fontsize=8)
    return fig


# =============================================================================
# SECTION 9: BASIN OF ATTRACTION ANALYSIS
# =============================================================================

def classify_trajectory_fate(system: dcABPFlatWall,
                              initial_condition: Tuple[float, float],
                              t_max: float = 200,
                              tolerance: float = 0.1) -> Dict:
    t, x, theta = integrate_trajectory(system, initial_condition, (0, t_max), 2000)
    
    result = {
        'initial': initial_condition,
        'final_position': (x[-1], theta[-1]),
        'fate': 'unknown',
        'touches_wall': False,
        'converged_to': None,
        'min_x': np.min(x),
        'max_x': np.max(x)
    }
    
    # Check if trajectory touched the wall (x < 0)
    result['touches_wall'] = np.any(x < 0)
    
    # Get fixed points
    fp_results = stability_analysis(system, verbose=False)
    stable_fps = [(fp['x_star'], fp['theta_star']) for fp in fp_results 
                  if fp['valid'] and fp['stability'] == 'Stable']
    saddle_fps = [(fp['x_star'], fp['theta_star']) for fp in fp_results 
                  if fp['valid'] and fp['stability'] != 'Stable']
    
    # Check convergence to stable fixed point
    final_x, final_theta = x[-1], theta[-1] % (2*np.pi)
    
    for i, (x_fp, theta_fp) in enumerate(stable_fps):
        theta_fp_mod = theta_fp % (2*np.pi)
        dx = abs(final_x - x_fp)
        dtheta = min(abs(final_theta - theta_fp_mod), 
                     2*np.pi - abs(final_theta - theta_fp_mod))
        
        if dx < tolerance and dtheta < tolerance:
            result['fate'] = 'stable_fp'
            result['converged_to'] = f'stable_{i}'
            return result
    
    # Check if near saddle
    for i, (x_fp, theta_fp) in enumerate(saddle_fps):
        theta_fp_mod = theta_fp % (2*np.pi)
        dx = abs(final_x - x_fp)
        dtheta = min(abs(final_theta - theta_fp_mod),
                     2*np.pi - abs(final_theta - theta_fp_mod))
        
        if dx < tolerance and dtheta < tolerance:
            result['fate'] = 'near_saddle'
            result['converged_to'] = f'saddle_{i}'
            return result
    
    # Check for escape
    if x[-1] > 1.0:  # Moved far from wall
        result['fate'] = 'escapes'
        return result
    
    # Check for orbit behavior
    if len(x) > 100:
        x_late = x[-500:]
        theta_late = theta[-500:]
        x_range_val = np.max(x_late) - np.min(x_late)
        theta_range_val = np.max(theta_late) - np.min(theta_late)
        
        if x_range_val > 0.05 and theta_range_val > 0.1:
            result['fate'] = 'orbits'
            return result
    
    result['fate'] = 'unknown'
    return result


def compute_basin_of_attraction(system: dcABPFlatWall,
                                 x_range: Tuple[float, float] = (-1.5, 0.5),
                                 theta_range: Tuple[float, float] = (0, 2*np.pi),
                                 n_x: int = 30,
                                 n_theta: int = 30,
                                 t_max: float = 200,
                                 verbose: bool = True) -> Dict:
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta, endpoint=False)
    
    fate_grid = np.empty((n_x, n_theta), dtype=object)
    touches_wall_grid = np.zeros((n_x, n_theta), dtype=bool)
    
    total = n_x * n_theta
    count = 0
    
    if verbose:
        print(f"\nComputing basin of attraction (Flat Wall)...")
        print(f"Grid: {n_x} x {n_theta} = {total} trajectories")
    
    for i, x0 in enumerate(x_vals):
        for j, theta0 in enumerate(theta_vals):
            result = classify_trajectory_fate(system, (x0, theta0), t_max)
            fate_grid[i, j] = result['fate']
            touches_wall_grid[i, j] = result['touches_wall']
            
            count += 1
            if verbose and count % 50 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    # Compute statistics
    fate_counts = {}
    for fate in np.unique(fate_grid):
        fate_counts[fate] = np.sum(fate_grid == fate)
    
    wall_touch_count = np.sum(touches_wall_grid)
    wall_touch_to_stable = np.sum((touches_wall_grid) & (fate_grid == 'stable_fp'))
    
    if verbose:
        print(f"\nResults:")
        print(f"  Fate distribution:")
        for fate, cnt in fate_counts.items():
            print(f"    {fate}: {cnt} ({100*cnt/total:.1f}%)")
        print(f"  Trajectories touching wall: {wall_touch_count}")
        print(f"  Wall-touching -> stable FP: {wall_touch_to_stable}")
    
    return {
        'x_vals': x_vals,
        'theta_vals': theta_vals,
        'fate_grid': fate_grid,
        'touches_wall_grid': touches_wall_grid,
        'fate_counts': fate_counts,
        'wall_touch_count': wall_touch_count,
        'wall_touch_to_stable': wall_touch_to_stable,
        'system': system
    }


def plot_basin_of_attraction(basin_data: Dict,
                              figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    x_vals = basin_data['x_vals']
    theta_vals = basin_data['theta_vals']
    fate_grid = basin_data['fate_grid']
    touches_wall_grid = basin_data['touches_wall_grid']
    system = basin_data['system']
    
    X, THETA = np.meshgrid(x_vals, theta_vals, indexing='ij')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Map fates to numbers
    fate_map = {
        'stable_fp': 0,
        'near_saddle': 1,
        'escapes': 2,
        'orbits': 3,
        'unknown': 4
    }
    
    fate_numeric = np.zeros_like(X)
    for i in range(len(x_vals)):
        for j in range(len(theta_vals)):
            fate_numeric[i, j] = fate_map.get(fate_grid[i, j], 4)
    
    # Left plot: Fate classification
    cmap = plt.cm.colors.ListedColormap(['green', 'orange', 'blue', 'purple', 'gray'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    c = ax1.pcolormesh(X, THETA, fate_numeric, cmap=cmap, norm=norm)
    ax1.axvline(x=0, color='black', linewidth=2, label='Wall')
    ax1.axvspan(x_vals[0], 0, alpha=0.2, color='gray')
    
    # Mark fixed points
    fp_results = stability_analysis(system, verbose=False)
    for fp in fp_results:
        if fp['valid']:
            marker = 'o' if fp['stability'] == 'Stable' else 'x'
            color = 'lime' if fp['stability'] == 'Stable' else 'red'
            ax1.plot(fp['x_star'], fp['theta_star'], marker, markersize=15,
                     markeredgewidth=3, color=color, zorder=10)
    
    ax1.set_xlabel(r'$\tilde{x}$', fontsize=12)
    ax1.set_ylabel(r'$\theta$', fontsize=12)
    ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax1.set_title('Basin of Attraction: Trajectory Fates', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='→ Stable FP'),
        Patch(facecolor='orange', label='→ Near Saddle'),
        Patch(facecolor='blue', label='Escapes'),
        Patch(facecolor='purple', label='Orbits'),
        Patch(facecolor='gray', label='Unknown'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Wall-touching trajectories
    wall_touch_fate = np.zeros_like(X)
    for i in range(len(x_vals)):
        for j in range(len(theta_vals)):
            if not touches_wall_grid[i, j]:
                wall_touch_fate[i, j] = 0
            elif fate_grid[i, j] == 'stable_fp':
                wall_touch_fate[i, j] = 1
            else:
                wall_touch_fate[i, j] = 2
    
    cmap2 = plt.cm.colors.ListedColormap(['lightgray', 'green', 'red'])
    bounds2 = [-0.5, 0.5, 1.5, 2.5]
    norm2 = plt.cm.colors.BoundaryNorm(bounds2, cmap2.N)
    
    ax2.pcolormesh(X, THETA, wall_touch_fate, cmap=cmap2, norm=norm2)
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.axvspan(x_vals[0], 0, alpha=0.2, color='gray')
    
    for fp in fp_results:
        if fp['valid']:
            marker = 'o' if fp['stability'] == 'Stable' else 'x'
            color = 'lime' if fp['stability'] == 'Stable' else 'darkred'
            ax2.plot(fp['x_star'], fp['theta_star'], marker, markersize=15,
                     markeredgewidth=3, color=color, zorder=10)
    
    ax2.set_xlabel(r'$\tilde{x}$', fontsize=12)
    ax2.set_ylabel(r'$\theta$', fontsize=12)
    ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax2.set_title('Wall-Touching Trajectories', fontsize=12)
    
    legend_elements2 = [
        Patch(facecolor='lightgray', label="Doesn't touch wall"),
        Patch(facecolor='green', label='Touches wall → Stable FP'),
        Patch(facecolor='red', label='Touches wall → Other'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    total = len(x_vals) * len(theta_vals)
    wall_touch_pct = 100 * basin_data['wall_touch_count'] / total
    wall_to_stable_pct = 100 * basin_data['wall_touch_to_stable'] / max(1, basin_data['wall_touch_count'])
    
    fig.suptitle(f"Basin of Attraction (Flat Wall)\n" +
                 f"$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, $\\tilde{{k}}$ = {system.k_tilde:.2f}\n" +
                 f"Wall-touching: {wall_touch_pct:.1f}%, of which {wall_to_stable_pct:.1f}% → Stable FP",
                 fontsize=12, y=1.05)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 10: sin(theta*) ANALYSIS
# =============================================================================

def analyze_sin_theta_flat_wall(omega_range: Tuple[float, float] = (-0.99, -0.01),
                                 k_range: Tuple[float, float] = (0.1, 5.0),
                                 n_omega: int = 50,
                                 n_k: int = 50,
                                 verbose: bool = False) -> Dict:
    omega_vals = np.linspace(omega_range[0], omega_range[1], n_omega)
    k_vals = np.linspace(k_range[0], k_range[1], n_k)
    
    plus_sin = np.full((n_omega, n_k), np.nan)
    plus_stability = np.full((n_omega, n_k), np.nan)
    minus_sin = np.full((n_omega, n_k), np.nan)
    minus_stability = np.full((n_omega, n_k), np.nan)
    n_fixed_points = np.zeros((n_omega, n_k))
    
    for i, omega in enumerate(omega_vals):
        for j, k in enumerate(k_vals):
            system = dcABPFlatWall(omega, k)
            results = stability_analysis(system, verbose=False)
            
            n_valid = 0
            for fp in results:
                if fp['valid']:
                    n_valid += 1
                    if fp['branch'] == 'plus':
                        plus_sin[i, j] = fp['sin_theta_star']
                        plus_stability[i, j] = 1 if fp['stability'] == 'Stable' else 0
                    else:
                        minus_sin[i, j] = fp['sin_theta_star']
                        minus_stability[i, j] = 1 if fp['stability'] == 'Stable' else 0
            
            n_fixed_points[i, j] = n_valid
    
    return {
        'omega_vals': omega_vals,
        'k_vals': k_vals,
        'plus_sin': plus_sin,
        'plus_stability': plus_stability,
        'minus_sin': minus_sin,
        'minus_stability': minus_stability,
        'n_fixed_points': n_fixed_points
    }


def plot_sin_theta_analysis_flat_wall(analysis_data: Dict,
                                       figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    omega = analysis_data['omega_vals']
    k = analysis_data['k_vals']
    OMEGA, K = np.meshgrid(omega, k, indexing='ij')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # sin(theta*) for plus branch
    ax = axes[0, 0]
    c = ax.pcolormesh(OMEGA, K, analysis_data['plus_sin'], cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title(r"$\sin(\theta^*)$ for '+' branch (sin > 0)", fontsize=12)
    fig.colorbar(c, ax=ax, label=r'$\sin(\theta^*)$')
    
    # sin(theta*) for minus branch
    ax = axes[0, 1]
    c = ax.pcolormesh(OMEGA, K, analysis_data['minus_sin'], cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title(r"$\sin(\theta^*)$ for '-' branch (sin < 0)", fontsize=12)
    fig.colorbar(c, ax=ax, label=r'$\sin(\theta^*)$')
    
    # Stability for plus branch
    ax = axes[1, 0]
    c = ax.pcolormesh(OMEGA, K, analysis_data['plus_stability'], cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title("'+' branch stability (green=stable)", fontsize=12)
    fig.colorbar(c, ax=ax, label='Stability')
    
    # Stability for minus branch
    ax = axes[1, 1]
    c = ax.pcolormesh(OMEGA, K, analysis_data['minus_stability'], cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title("'-' branch stability (green=stable)", fontsize=12)
    fig.colorbar(c, ax=ax, label='Stability')
    
    fig.suptitle(r"$\sin(\theta^*)$ Analysis for Flat Wall", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 11: TWO-PARAMETER PHASE DIAGRAM
# =============================================================================

def compute_phase_diagram_flat_wall(omega_range: Tuple[float, float] = (-1.5, 0.5),
                                     k_range: Tuple[float, float] = (0.1, 5.0),
                                     n_omega: int = 100,
                                     n_k: int = 100) -> Dict:
    omega_vals = np.linspace(omega_range[0], omega_range[1], n_omega)
    k_vals = np.linspace(k_range[0], k_range[1], n_k)
    
    n_fixed_points = np.zeros((n_omega, n_k))
    has_stable = np.zeros((n_omega, n_k))
    has_saddle = np.zeros((n_omega, n_k))
    stable_sin_sign = np.full((n_omega, n_k), np.nan)
    saddle_sin_sign = np.full((n_omega, n_k), np.nan)
    
    for i, omega in enumerate(omega_vals):
        for j, k in enumerate(k_vals):
            system = dcABPFlatWall(omega, k)
            results = stability_analysis(system, verbose=False)
            
            n_valid = 0
            for fp in results:
                if fp['valid']:
                    n_valid += 1
                    if fp['stability'] == 'Stable':
                        has_stable[i, j] = 1
                        stable_sin_sign[i, j] = np.sign(fp['sin_theta_star'])
                    else:
                        has_saddle[i, j] = 1
                        saddle_sin_sign[i, j] = np.sign(fp['sin_theta_star'])
            
            n_fixed_points[i, j] = n_valid
    
    return {
        'omega_vals': omega_vals,
        'k_vals': k_vals,
        'n_fixed_points': n_fixed_points,
        'has_stable': has_stable,
        'has_saddle': has_saddle,
        'stable_sin_sign': stable_sin_sign,
        'saddle_sin_sign': saddle_sin_sign
    }


def plot_phase_diagram_flat_wall(phase_data: Dict,
                                  figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    omega = phase_data['omega_vals']
    k = phase_data['k_vals']
    OMEGA, K = np.meshgrid(omega, k, indexing='ij')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Number of fixed points
    ax = axes[0, 0]
    c = ax.pcolormesh(OMEGA, K, phase_data['n_fixed_points'], cmap='viridis', vmin=0, vmax=2)
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label=r'$\tilde{\omega}=0$')
    ax.axvline(x=-1, color='red', linewidth=2, linestyle='--', label=r'$\tilde{\omega}=-1$')
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title('Number of Fixed Points', fontsize=12)
    fig.colorbar(c, ax=ax, label='# fixed points')
    ax.legend(loc='upper right')
    
    # Annotate regions
    ax.text(-0.5, 2.5, '2 fixed points\n(stable + saddle)', fontsize=10,
            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.25, 2.5, 'No fixed\npoints', fontsize=10,
            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Phase regions
    ax = axes[0, 1]
    phase_regions = (phase_data['has_stable'] + 2*phase_data['has_saddle']).astype(int)
    cmap = plt.cm.colors.ListedColormap(['white', 'green', 'red', 'gold'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    ax.pcolormesh(OMEGA, K, phase_regions, cmap=cmap, norm=norm)
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.axvline(x=-1, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title('Fixed Point Types', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='white', edgecolor='black', label='No fixed points'),
                       Patch(facecolor='gold', label='Stable + Saddle')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # sin(theta*) at stable FP
    ax = axes[1, 0]
    c = ax.pcolormesh(OMEGA, K, phase_data['stable_sin_sign'], cmap='RdBu', vmin=-1, vmax=1)
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.axvline(x=-1, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title(r'Sign of $\sin(\theta^*)$ at STABLE FP', fontsize=12)
    fig.colorbar(c, ax=ax, label=r'sign($\sin\theta^*$)')
    
    # sin(theta*) at saddle FP
    ax = axes[1, 1]
    c = ax.pcolormesh(OMEGA, K, phase_data['saddle_sin_sign'], cmap='RdBu', vmin=-1, vmax=1)
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.axvline(x=-1, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\tilde{k}$', fontsize=12)
    ax.set_title(r'Sign of $\sin(\theta^*)$ at SADDLE FP', fontsize=12)
    fig.colorbar(c, ax=ax, label=r'sign($\sin\theta^*$)')
    
    fig.suptitle(r"Two-Parameter Phase Diagram (Flat Wall) in $(\tilde{\omega}, \tilde{k})$ space",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("dcABP Flat Wall Analysis - Complete Module")
    print("="*70)
    print("\nThis module provides comprehensive flat wall analysis functions.")
    print("Import it in your playground file to use the functions.")
    print("\nMain functions available:")
    print("  - dcABPFlatWall: System class")
    print("  - find_fixed_points_detailed: Detailed fixed point analysis")
    print("  - stability_analysis: Full stability analysis")
    print("  - plot_phase_portrait: Phase portrait plotting")
    print("  - parameter_sweep_omega/k: Parameter sweeps")
    print("  - bifurcation_diagram: Bifurcation diagrams")
    print("  - compute_basin_of_attraction: Basin of attraction analysis")
    print("  - analyze_sin_theta_flat_wall: sin(theta*) systematic analysis")
    print("  - compute_phase_diagram_flat_wall: Two-parameter phase diagram")
    print("  - print_theoretical_summary: Print theory summary")
