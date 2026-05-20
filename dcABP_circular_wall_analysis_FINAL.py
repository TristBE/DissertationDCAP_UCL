
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from typing import Tuple, Optional, List, Dict
import warnings

# =============================================================================
# SECTION 1: SYSTEM DEFINITION
# =============================================================================

class dcABPCircularWall:
    
    def __init__(self, omega_tilde: float = -0.5, R_w_tilde: float = 5.0, 
                 k_tilde: float = 10.0, confinement: str = 'inside'):
        self.omega_tilde = omega_tilde
        self.R_w_tilde = R_w_tilde
        self.k_tilde = k_tilde
        self.confinement = confinement
        
        if confinement not in ['inside', 'outside']:
            raise ValueError("confinement must be 'inside' or 'outside'")
        if R_w_tilde <= 0:
            raise ValueError("R_w_tilde must be positive")
    
    def wall_force(self, r: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=float))
        F = np.zeros_like(r)
        
        if self.confinement == 'inside':
            mask = r > self.R_w_tilde
            F[mask] = -self.k_tilde * (r[mask] - self.R_w_tilde)
        else:  # outside
            mask = r < self.R_w_tilde
            F[mask] = -self.k_tilde * (r[mask] - self.R_w_tilde)
        
        return F
    
    def wall_force_derivative(self, r: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=float))
        dF = np.zeros_like(r)
        
        if self.confinement == 'inside':
            dF[r > self.R_w_tilde] = -self.k_tilde
        else:  # outside
            dF[r < self.R_w_tilde] = -self.k_tilde
        
        return dF
    
    def dr_dt(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return np.cos(theta) + self.wall_force(r)
    
    def dtheta_dt(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        
        # Avoid division by zero at r = 0
        r_safe = np.where(r > 1e-10, r, 1e-10)
        
        return (1 + self.omega_tilde + 
                self.wall_force(r) * np.cos(theta) - 
                np.sin(theta) / r_safe)
    
    def dphi_dt(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(r, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        r_safe = np.where(r > 1e-10, r, 1e-10)
        return np.sin(theta) / r_safe
    
    def vector_field(self, r: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.dr_dt(r, theta), self.dtheta_dt(r, theta)


# =============================================================================
# SECTION 2: FIXED POINT ANALYSIS
# =============================================================================

def find_fixed_points(system: dcABPCircularWall, 
                      verbose: bool = False) -> List[Tuple[float, float]]:
    omega = system.omega_tilde
    R_w = system.R_w_tilde
    k = system.k_tilde
    
    fixed_points = []
    
    # Check existence condition
    discriminant = 1/(4 * R_w**2) - omega
    
    if verbose:
        print(f"\nExistence condition: omega~ <= 1/(4*R~_w^2) = {1/(4*R_w**2):.6f}")
        print(f"Current omega~ = {omega:.6f}")
        print(f"Discriminant = {discriminant:.6f}")
    
    if discriminant < 0:
        if verbose:
            print("No fixed points exist (discriminant < 0)")
        return fixed_points
    
    # Calculate sin(theta*) values
    sqrt_disc = np.sqrt(discriminant)
    s_plus = 1/(2*R_w) + sqrt_disc
    s_minus = 1/(2*R_w) - sqrt_disc
    
    if verbose:
        print(f"\nsin(theta*) solutions:")
        print(f"  s+ = 1/(2*R~_w) + sqrt(disc) = {s_plus:.6f}")
        print(f"  s- = 1/(2*R~_w) - sqrt(disc) = {s_minus:.6f}")
    
    # Process each solution
    for s in [s_plus, s_minus]:
        # Check validity
        if not (-1 <= s <= 1):
            if verbose:
                print(f"  |s| = {abs(s):.4f} > 1, skipping")
            continue
        
        # Calculate theta* based on confinement type
        if system.confinement == 'inside':
            # Need cos(theta*) > 0, so theta* in (-pi/2, pi/2)
            theta_star = np.arcsin(s)
            if theta_star < 0:
                theta_star += 2*np.pi
        else:
            # Need cos(theta*) < 0, so theta* in (pi/2, 3*pi/2)
            theta_star = np.pi - np.arcsin(s)
        
        cos_theta = np.cos(theta_star)
        r_star = R_w + cos_theta / k
        
        if verbose:
            print(f"\nFixed point found:")
            print(f"  sin(theta*) = {s:.6f}")
            print(f"  theta* = {theta_star:.4f} rad = {np.degrees(theta_star):.2f} deg")
            print(f"  cos(theta*) = {cos_theta:.6f}")
            print(f"  r* = R~_w + cos(theta*)/k~ = {r_star:.4f}")
        
        fixed_points.append((r_star, theta_star))
    
    return fixed_points


# =============================================================================
# SECTION 3: JACOBIAN AND STABILITY ANALYSIS
# =============================================================================

def compute_jacobian(system: dcABPCircularWall, 
                     r: float, theta: float) -> np.ndarray:
    r_arr = np.array([r])
    F = system.wall_force(r_arr)[0]
    dF = system.wall_force_derivative(r_arr)[0]
    
    J = np.array([
        [dF, -np.sin(theta)],
        [dF * np.cos(theta) + np.sin(theta)/r**2, -F * np.sin(theta) - np.cos(theta)/r]
    ])
    
    return J


def classify_fixed_point(jacobian: np.ndarray) -> dict:
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
    
    # Check if eigenvalues are complex
    if np.iscomplex(eigenvalues).any() and np.abs(eigenvalues[0].imag) > 1e-10:
        real_part = eigenvalues[0].real
        if real_part < -1e-10:
            result['classification'] = 'Stable Focus'
            result['stability'] = 'Stable'
        elif real_part > 1e-10:
            result['classification'] = 'Unstable Focus'
            result['stability'] = 'Unstable'
        else:
            result['classification'] = 'Center'
            result['stability'] = 'Marginal'
    else:
        # Real eigenvalues
        eig_real = np.real(eigenvalues)
        if np.all(eig_real < -1e-10):
            result['classification'] = 'Stable Node'
            result['stability'] = 'Stable'
        elif np.all(eig_real > 1e-10):
            result['classification'] = 'Unstable Node'
            result['stability'] = 'Unstable'
        elif eig_real[0] * eig_real[1] < 0:
            result['classification'] = 'Saddle Point'
            result['stability'] = 'Unstable'
        else:
            result['classification'] = 'Degenerate'
            result['stability'] = 'Marginal'
    
    return result


def stability_analysis(system: dcABPCircularWall, 
                       verbose: bool = True) -> List[dict]:
    if verbose:
        print("\n" + "="*70)
        print("STABILITY ANALYSIS - dcABP Circular Wall System")
        print("="*70)
        print(f"Parameters: omega~ = {system.omega_tilde:.4f}, "
              f"R~_w = {system.R_w_tilde:.4f}, k~ = {system.k_tilde:.4f}")
        print(f"Confinement: {system.confinement.upper()}")
        print("-"*70)
        print(f"Existence condition: omega~ <= 1/(4*R~_w^2) = {1/(4*system.R_w_tilde**2):.4f}")
        print(f"Current omega~ = {system.omega_tilde:.4f} -> Fixed points "
              f"{'CAN' if system.omega_tilde <= 1/(4*system.R_w_tilde**2) else 'CANNOT'} exist")
        print("-"*70)
    
    fixed_points = find_fixed_points(system)
    results = []
    
    for i, (r_fp, theta_fp) in enumerate(fixed_points):
        J = compute_jacobian(system, r_fp, theta_fp)
        result = classify_fixed_point(J)
        result['position'] = (r_fp, theta_fp)
        result['jacobian'] = J
        results.append(result)
        
        if verbose:
            print(f"\nFixed Point {i+1}:")
            print(f"  Position: (r*, theta*) = ({r_fp:.4f}, {theta_fp:.4f} rad)")
            print(f"           theta* = {np.degrees(theta_fp):.2f} deg")
            print(f"           r* - R~_w = {r_fp - system.R_w_tilde:.4f} (wall penetration)")
            print(f"  sin(theta*) = {np.sin(theta_fp):.4f}, cos(theta*) = {np.cos(theta_fp):.4f}")
            print(f"  Jacobian:")
            print(f"    [{J[0,0]:10.4f}  {J[0,1]:10.4f}]")
            print(f"    [{J[1,0]:10.4f}  {J[1,1]:10.4f}]")
            print(f"  Trace = {result['trace']:.4f}")
            print(f"  Determinant = {result['determinant']:.4f}")
            print(f"  Eigenvalues: {result['eigenvalues']}")
            print(f"  Classification: {result['classification']}")
            print(f"  Stability: {result['stability']}")
    
    return results


# =============================================================================
# SECTION 4: NULLCLINE COMPUTATION
# =============================================================================

def compute_r_nullcline(system: dcABPCircularWall, 
                        theta_range: np.ndarray) -> np.ndarray:
    return system.R_w_tilde + np.cos(theta_range) / system.k_tilde


def compute_theta_nullcline(system: dcABPCircularWall, 
                            r_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta_null_1 = np.full_like(r_range, np.nan)
    theta_null_2 = np.full_like(r_range, np.nan)
    
    for i, r in enumerate(r_range):
        F = system.wall_force(np.array([r]))[0]
        
        def equation(theta):
            return 1 + system.omega_tilde + F * np.cos(theta) - np.sin(theta)/r
        
        # Search for solutions in [0, 2*pi]
        solutions = []
        for theta_init in np.linspace(0.1, 2*np.pi - 0.1, 20):
            try:
                sol = fsolve(equation, theta_init, full_output=True)
                if sol[2] == 1:  # Converged
                    theta_sol = sol[0][0] % (2*np.pi)
                    if abs(equation(theta_sol)) < 1e-6:
                        # Check if this is a new solution
                        is_new = True
                        for existing in solutions:
                            if abs(theta_sol - existing) < 0.1:
                                is_new = False
                                break
                        if is_new:
                            solutions.append(theta_sol)
            except:
                pass
        
        solutions.sort()
        if len(solutions) >= 1:
            theta_null_1[i] = solutions[0]
        if len(solutions) >= 2:
            theta_null_2[i] = solutions[1]
    
    return theta_null_1, theta_null_2


# =============================================================================
# SECTION 5: PHASE PORTRAIT PLOTTING (CARTESIAN)
# =============================================================================

def plot_phase_portrait(system: dcABPCircularWall,
                        r_range: Optional[Tuple[float, float]] = None,
                        theta_range: Tuple[float, float] = (0, 2*np.pi),
                        resolution: int = 25,
                        figsize: Tuple[int, int] = (10, 8),
                        show_nullclines: bool = True,
                        show_fixed_points: bool = True,
                        title: Optional[str] = None) -> plt.Figure:
    R_w = system.R_w_tilde
    
    # Set default r_range based on confinement type
    if r_range is None:
        if system.confinement == 'inside':
            r_range = (max(0.5, R_w - 1), R_w + 2)
        else:
            r_range = (max(0.5, R_w - 2), R_w + 1)
    
    # Create meshgrid
    r = np.linspace(r_range[0], r_range[1], resolution)
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    R, THETA = np.meshgrid(r, theta)
    
    # Compute vector field
    dR, dTHETA = system.vector_field(R, THETA)
    
    # Compute speed for coloring
    speed = np.sqrt(dR**2 + dTHETA**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot streamlines
    strm = ax.streamplot(R, THETA, dR, dTHETA,
                         color=speed, cmap='viridis',
                         density=1.5, linewidth=0.8,
                         arrowsize=1.2)
    
    # Add colorbar
    cbar = fig.colorbar(strm.lines, ax=ax, label='|velocity|')
    
    # Plot wall location
    ax.axvline(x=R_w, color='blue', linewidth=2, linestyle='-', 
               label=f'Wall (r~ = {R_w:.1f})')
    
    # Shade the wall region
    if system.confinement == 'inside':
        ax.axvspan(R_w, r_range[1], alpha=0.15, color='blue', label='Wall region')
    else:
        ax.axvspan(r_range[0], R_w, alpha=0.15, color='blue', label='Wall region')
    
    # Plot nullclines
    if show_nullclines:
        # r-nullcline: dr~/dt~ = 0
        theta_fine = np.linspace(theta_range[0], theta_range[1], 500)
        r_null = compute_r_nullcline(system, theta_fine)
        
        # Only plot in valid region
        if system.confinement == 'inside':
            mask = (r_null >= R_w) & (r_null <= r_range[1])
        else:
            mask = (r_null >= r_range[0]) & (r_null <= R_w)
        
        ax.plot(r_null[mask], theta_fine[mask], 'r-', linewidth=2, 
                label=r'$\dot{r}=0$ nullcline')
        
        # theta-nullcline: dtheta/dt~ = 0 (compute numerically)
        r_fine = np.linspace(r_range[0], r_range[1], 100)
        theta_null_1, theta_null_2 = compute_theta_nullcline(system, r_fine)
        
        valid_1 = ~np.isnan(theta_null_1)
        valid_2 = ~np.isnan(theta_null_2)
        
        if np.any(valid_1):
            ax.plot(r_fine[valid_1], theta_null_1[valid_1], 'b-', linewidth=2,
                    label=r'$\dot{\theta}=0$ nullcline')
        if np.any(valid_2):
            ax.plot(r_fine[valid_2], theta_null_2[valid_2], 'b-', linewidth=2)
    
    # Mark fixed points
    if show_fixed_points:
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                marker = 'o'
                color = 'green'
            elif result['stability'] == 'Unstable' and 'Saddle' in result['classification']:
                marker = 'x'
                color = 'red'
            else:
                marker = 's'
                color = 'orange'
            
            ax.plot(r_fp, theta_fp, marker, markersize=8, 
                    markeredgewidth=2, color=color, alpha=0.6,
                    label=f"{result['classification']} ({r_fp:.2f}, {theta_fp:.2f})")
    
    # Labels and formatting
    ax.set_xlabel(r'$\tilde{r}$', fontsize=12)
    ax.set_ylabel(r'$\theta$', fontsize=12)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    if title is None:
        title = (f'Phase Portrait ({system.confinement.upper()} confinement)\n'
                f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, '
                f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f}, '
                f'$\\tilde{{k}}$ = {system.k_tilde:.1f}')
    ax.set_title(title, fontsize=12)
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6: PARAMETER SWEEP PLOTS
# =============================================================================

def parameter_sweep_omega(R_w_tilde: float, k_tilde: float = 10.0,
                          omega_values: List[float] = None,
                          confinement: str = 'inside',
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if omega_values is None:
        omega_values = [-0.9, -0.5, -0.2, -0.05]
    
    n_plots = len(omega_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, omega in enumerate(omega_values):
        ax = axes[idx]
        system = dcABPCircularWall(omega, R_w_tilde, k_tilde, confinement)
        
        if confinement == 'inside':
            r_range = (max(0.5, R_w_tilde - 1), R_w_tilde + 2)
        else:
            r_range = (max(0.5, R_w_tilde - 2), R_w_tilde + 1)
        
        r = np.linspace(r_range[0], r_range[1], 25)
        theta = np.linspace(0, 2*np.pi, 25)
        R, THETA = np.meshgrid(r, theta)
        dR, dTHETA = system.vector_field(R, THETA)
        speed = np.sqrt(dR**2 + dTHETA**2)
        
        ax.streamplot(R, THETA, dR, dTHETA, color=speed, cmap='viridis',
                     density=1.2, linewidth=0.6, arrowsize=1.0)
        ax.axvline(x=R_w_tilde, color='blue', linewidth=2)
        
        if confinement == 'inside':
            ax.axvspan(R_w_tilde, r_range[1], alpha=0.15, color='blue')
        else:
            ax.axvspan(r_range[0], R_w_tilde, alpha=0.15, color='blue')
        
        # Mark fixed points
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            marker = 'o' if result['stability'] == 'Stable' else 'x'
            color = 'green' if result['stability'] == 'Stable' else 'red'
            ax.plot(r_fp, theta_fp, marker, markersize=10, markeredgewidth=2, color=color)
        
        ax.set_xlabel(r'$\tilde{r}$', fontsize=10)
        ax.set_ylabel(r'$\theta$', fontsize=10)
        ax.set_title(f'$\\tilde{{\\omega}}$ = {omega}', fontsize=11)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Parameter sweep: $\\tilde{{\\omega}}$ ({confinement.upper()} confinement)\n'
                f'$\\tilde{{R}}_w$ = {R_w_tilde}, $\\tilde{{k}}$ = {k_tilde}', fontsize=13)
    plt.tight_layout()
    return fig


def parameter_sweep_R_w(omega_tilde: float, k_tilde: float = 10.0,
                        R_w_values: List[float] = None,
                        confinement: str = 'inside',
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if R_w_values is None:
        R_w_values = [1.0, 2.0, 5.0, 20.0]
    
    n_plots = len(R_w_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, R_w in enumerate(R_w_values):
        ax = axes[idx]
        system = dcABPCircularWall(omega_tilde, R_w, k_tilde, confinement)
        
        if confinement == 'inside':
            r_range = (max(0.5, R_w - 1), R_w + 2)
        else:
            r_range = (max(0.5, R_w - 2), R_w + 1)
        
        r = np.linspace(r_range[0], r_range[1], 25)
        theta = np.linspace(0, 2*np.pi, 25)
        R, THETA = np.meshgrid(r, theta)
        dR, dTHETA = system.vector_field(R, THETA)
        speed = np.sqrt(dR**2 + dTHETA**2)
        
        ax.streamplot(R, THETA, dR, dTHETA, color=speed, cmap='viridis',
                     density=1.2, linewidth=0.6, arrowsize=1.0)
        ax.axvline(x=R_w, color='blue', linewidth=2)
        
        if confinement == 'inside':
            ax.axvspan(R_w, r_range[1], alpha=0.15, color='blue')
        else:
            ax.axvspan(r_range[0], R_w, alpha=0.15, color='blue')
        
        # Mark fixed points
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            marker = 'o' if result['stability'] == 'Stable' else 'x'
            color = 'green' if result['stability'] == 'Stable' else 'red'
            ax.plot(r_fp, theta_fp, marker, markersize=10, markeredgewidth=2, color=color)
        
        ax.set_xlabel(r'$\tilde{r}$', fontsize=10)
        ax.set_ylabel(r'$\theta$', fontsize=10)
        ax.set_title(f'$\\tilde{{R}}_w$ = {R_w}', fontsize=11)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Parameter sweep: $\\tilde{{R}}_w$ ({confinement.upper()} confinement)\n'
                f'$\\tilde{{\\omega}}$ = {omega_tilde}, $\\tilde{{k}}$ = {k_tilde}', fontsize=13)
    plt.tight_layout()
    return fig


def parameter_sweep_k(omega_tilde: float, R_w_tilde: float,
                      k_values: List[float] = None,
                      confinement: str = 'inside',
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if k_values is None:
        k_values = [1.0, 5.0, 10.0, 50.0]
    
    n_plots = len(k_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        system = dcABPCircularWall(omega_tilde, R_w_tilde, k, confinement)
        
        # Adjust r_range based on expected penetration depth
        penetration = 2.0 / k
        if confinement == 'inside':
            r_range = (max(0.5, R_w_tilde - 1), R_w_tilde + penetration + 1)
        else:
            r_range = (max(0.5, R_w_tilde - penetration - 1), R_w_tilde + 1)
        
        r = np.linspace(r_range[0], r_range[1], 25)
        theta = np.linspace(0, 2*np.pi, 25)
        R, THETA = np.meshgrid(r, theta)
        dR, dTHETA = system.vector_field(R, THETA)
        speed = np.sqrt(dR**2 + dTHETA**2)
        
        ax.streamplot(R, THETA, dR, dTHETA, color=speed, cmap='viridis',
                     density=1.2, linewidth=0.6, arrowsize=1.0)
        ax.axvline(x=R_w_tilde, color='blue', linewidth=2)
        
        if confinement == 'inside':
            ax.axvspan(R_w_tilde, r_range[1], alpha=0.15, color='blue')
        else:
            ax.axvspan(r_range[0], R_w_tilde, alpha=0.15, color='blue')
        
        # Mark fixed points
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            marker = 'o' if result['stability'] == 'Stable' else 'x'
            color = 'green' if result['stability'] == 'Stable' else 'red'
            ax.plot(r_fp, theta_fp, marker, markersize=10, markeredgewidth=2, color=color)
        
        ax.set_xlabel(r'$\tilde{r}$', fontsize=10)
        ax.set_ylabel(r'$\theta$', fontsize=10)
        ax.set_title(f'$\\tilde{{k}}$ = {k}', fontsize=11)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Parameter sweep: $\\tilde{{k}}$ ({confinement.upper()} confinement)\n'
                f'$\\tilde{{\\omega}}$ = {omega_tilde}, $\\tilde{{R}}_w$ = {R_w_tilde}', fontsize=13)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7: BIFURCATION DIAGRAMS
# =============================================================================

def bifurcation_diagram_omega(R_w_tilde: float, k_tilde: float = 10.0,
                               omega_range: Tuple[float, float] = (-0.99, 0.01),
                               confinement: str = 'inside',
                               n_points: int = 100,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    omega_vals = np.linspace(omega_range[0], omega_range[1], n_points)
    
    theta_stable = []
    theta_unstable = []
    omega_stable = []
    omega_unstable = []
    
    for omega in omega_vals:
        system = dcABPCircularWall(omega, R_w_tilde, k_tilde, confinement)
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                theta_stable.append(theta_fp)
                omega_stable.append(omega)
            else:
                theta_unstable.append(theta_fp)
                omega_unstable.append(omega)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if omega_stable:
        ax.scatter(omega_stable, theta_stable, c='green', s=10, label='Stable', alpha=0.7)
    if omega_unstable:
        ax.scatter(omega_unstable, theta_unstable, c='red', s=10, label='Unstable', alpha=0.7)
    
    # Mark critical omega value
    omega_crit = 1/(4 * R_w_tilde**2)
    ax.axvline(x=omega_crit, color='purple', linestyle='--', linewidth=2,
               label=f'$\\omega^*_{{crit}}$ = 1/(4$\\tilde{{R}}_w^2$) = {omega_crit:.4f}')
    
    ax.set_xlabel(r'$\tilde{\omega}$', fontsize=12)
    ax.set_ylabel(r'$\theta^*$', fontsize=12)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_title(f'Bifurcation Diagram: $\\theta^*$ vs $\\tilde{{\\omega}}$\n'
                f'({confinement.upper()}, $\\tilde{{R}}_w$ = {R_w_tilde})', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def bifurcation_diagram_R_w(omega_tilde: float, k_tilde: float = 10.0,
                             R_w_range: Tuple[float, float] = (0.5, 25.0),
                             confinement: str = 'inside',
                             n_points: int = 100,
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    R_w_vals = np.linspace(R_w_range[0], R_w_range[1], n_points)
    
    theta_stable = []
    theta_unstable = []
    R_w_stable = []
    R_w_unstable = []
    
    for R_w in R_w_vals:
        system = dcABPCircularWall(omega_tilde, R_w, k_tilde, confinement)
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                theta_stable.append(theta_fp)
                R_w_stable.append(R_w)
            else:
                theta_unstable.append(theta_fp)
                R_w_unstable.append(R_w)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if R_w_stable:
        ax.scatter(R_w_stable, theta_stable, c='green', s=10, label='Stable', alpha=0.7)
    if R_w_unstable:
        ax.scatter(R_w_unstable, theta_unstable, c='red', s=10, label='Unstable', alpha=0.7)
    
    # Show flat wall limit values
    if omega_tilde < 0:
        theta_flat_1 = np.arcsin(np.sqrt(-omega_tilde))
        theta_flat_2 = 2*np.pi - theta_flat_1
        ax.axhline(y=theta_flat_1, color='purple', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Flat wall limit ($\\tilde{{R}}_w \\to \\infty$)')
        ax.axhline(y=theta_flat_2, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel(r'$\tilde{R}_w$', fontsize=12)
    ax.set_ylabel(r'$\theta^*$', fontsize=12)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_title(f'Bifurcation Diagram: $\\theta^*$ vs $\\tilde{{R}}_w$\n'
                f'({confinement.upper()}, $\\tilde{{\\omega}}$ = {omega_tilde})', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 8: COMPARISON PLOTS
# =============================================================================

def compare_inside_outside(omega_tilde: float, R_w_tilde: float, k_tilde: float = 10.0,
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for ax, conf in [(ax1, 'inside'), (ax2, 'outside')]:
        system = dcABPCircularWall(omega_tilde, R_w_tilde, k_tilde, conf)
        
        if conf == 'inside':
            r_range = (R_w_tilde - 0.5, R_w_tilde + 1.5)
        else:
            r_range = (max(0.5, R_w_tilde - 1.5), R_w_tilde + 0.5)
        
        r = np.linspace(r_range[0], r_range[1], 30)
        theta = np.linspace(0, 2*np.pi, 30)
        R, THETA = np.meshgrid(r, theta)
        dR, dTHETA = system.vector_field(R, THETA)
        speed = np.sqrt(dR**2 + dTHETA**2)
        
        ax.streamplot(R, THETA, dR, dTHETA, color=speed, cmap='viridis',
                     density=1.5, linewidth=0.7, arrowsize=1.2)
        
        ax.axvline(x=R_w_tilde, color='blue', linewidth=2)
        if conf == 'inside':
            ax.axvspan(R_w_tilde, r_range[1], alpha=0.15, color='blue')
        else:
            ax.axvspan(r_range[0], R_w_tilde, alpha=0.15, color='blue')
        
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            marker = 'o' if result['stability'] == 'Stable' else 'x'
            color = 'green' if result['stability'] == 'Stable' else 'red'
            ax.plot(r_fp, theta_fp, marker, markersize=12, markeredgewidth=3, color=color)
        
        ax.set_xlabel(r'$\tilde{r}$', fontsize=12)
        ax.set_ylabel(r'$\theta$', fontsize=12)
        ax.set_title(f'{conf.upper()} confinement', fontsize=12)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'$\\tilde{{\\omega}}$ = {omega_tilde}, $\\tilde{{R}}_w$ = {R_w_tilde}, '
                f'$\\tilde{{k}}$ = {k_tilde}', fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 9: TRAJECTORY INTEGRATION
# =============================================================================


def integrate_full_trajectory(system,
                              initial_condition: Tuple[float, float, float],
                              t_span: Tuple[float, float] = (0, 100),
                              n_points: int = 2000):
    def rhs(t, y):
        r, theta, phi = y
        dr = system.dr_dt(np.array([r]), np.array([theta]))[0]
        dtheta = system.dtheta_dt(np.array([r]), np.array([theta]))[0]
        dphi = system.dphi_dt(np.array([r]), np.array([theta]))[0]
        return [dr, dtheta, dphi]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, initial_condition, t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def plot_cartesian_trajectory(system,
                              initial_condition: Tuple[float, float, float],
                              t_span: Tuple[float, float] = (0, 100),
                              zoom_padding: float = 1.5,
                              wall_depth: float = 2.0,
                              figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
    # Integrate
    t, r, theta, phi = integrate_full_trajectory(system, initial_condition, t_span)

    # Convert to Cartesian
    x = r * np.cos(phi)
    y_pos = r * np.sin(phi)

    fig, ax = plt.subplots(figsize=figsize)
    R_w = system.R_w_tilde

    # ── Wall shading (matches phase portrait axvspan style) ──────────
    n_ring = 200
    wall_theta = np.linspace(0, 2 * np.pi, n_ring)

    if system.confinement == 'inside':
        # Shade from R_w outward (the wall the particle pushes into)
        r_inner = R_w
        r_outer = R_w + wall_depth
    else:
        # Shade from R_w inward
        r_inner = max(0, R_w - wall_depth)
        r_outer = R_w

    # Fill the annular wall region
    theta_fill = np.linspace(0, 2 * np.pi, n_ring)
    x_inner = r_inner * np.cos(theta_fill)
    y_inner = r_inner * np.sin(theta_fill)
    x_outer = r_outer * np.cos(theta_fill)
    y_outer = r_outer * np.sin(theta_fill)

    ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
            np.concatenate([y_outer, y_inner[::-1]]),
            color='blue', alpha=0.15, label='Wall region')

    # Wall boundary line
    ax.plot(R_w * np.cos(wall_theta), R_w * np.sin(wall_theta),
            color='blue', linewidth=2,
            label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.1f})')

    # ── Trajectory ───────────────────────────────────────────────────
    ax.plot(x, y_pos, 'k-', linewidth=0.6, alpha=0.7)
    ax.plot(x[0], y_pos[0], 'o', color='green', markersize=10,
            markeredgecolor='black', markeredgewidth=1, label='Start', zorder=5)
    ax.plot(x[-1], y_pos[-1], 's', color='red', markersize=8,
            markeredgecolor='black', markeredgewidth=1, label='End', zorder=5)

    # ── Axis limits: zoom into the confinement region ────────────────
    if system.confinement == 'inside':
        lim = R_w + zoom_padding
    else:
        lim = R_w + zoom_padding

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    # ── Labels and formatting ────────────────────────────────────────
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


def plot_cartesian_trajectories(system,
                                initial_conditions: List[Tuple[float, float, float]],
                                t_span: Tuple[float, float] = (0, 100),
                                zoom_padding: float = 1.5,
                                wall_depth: float = 2.0,
                                figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    R_w = system.R_w_tilde

    # ── Wall shading ─────────────────────────────────────────────────
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

    # ── Trajectories ─────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))

    for ic, col in zip(initial_conditions, colors):
        t, r, theta, phi = integrate_full_trajectory(system, ic, t_span)
        x = r * np.cos(phi)
        y_pos = r * np.sin(phi)

        ax.plot(x, y_pos, '-', color=col, linewidth=0.6, alpha=0.8)
        ax.plot(x[0], y_pos[0], 'o', color=col, markersize=10,
                markeredgecolor='black', markeredgewidth=1,
                label=f'IC: ({ic[0]:.1f}, {ic[1]:.2f}, {ic[2]:.2f})',
                zorder=5)
        ax.plot(x[-1], y_pos[-1], 's', color=col, markersize=7,
                markeredgecolor='black', markeredgewidth=1, zorder=5)

    # ── Axis limits ──────────────────────────────────────────────────
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


# =============================================================================
# SECTION 10: POLAR PHASE PORTRAIT VISUALISATION
# =============================================================================

def plot_polar_phase_portrait(system: dcABPCircularWall,
                               r_range: Optional[Tuple[float, float]] = None,
                               n_theta_lines: int = 12,
                               n_r_lines: int = 8,
                               t_max: float = 30.0,
                               resolution: int = 500,
                               figsize: Tuple[int, int] = (10, 10),
                               show_fixed_points: bool = True,
                               show_wall: bool = True,
                               near_wall_zoom: bool = False,
                               title: Optional[str] = None) -> plt.Figure:
    R_w = system.R_w_tilde
    
    # Set default r_range based on confinement type
    if r_range is None:
        if near_wall_zoom:
            # Zoom into near-wall region (penetration depth ~ 1/k)
            penetration = 2.0 / system.k_tilde + 0.5  # Approximate penetration depth + margin
            if system.confinement == 'inside':
                r_range = (max(0.1, R_w - 0.5), R_w + penetration)
            else:
                r_range = (max(0.1, R_w - penetration), R_w + 0.5)
        else:
            if system.confinement == 'inside':
                r_range = (max(0.1, R_w - 1.0), R_w + 2.0)
            else:
                r_range = (max(0.1, R_w - 2.0), R_w + 1.0)
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    
    # Starting points for trajectories
    r_starts = np.linspace(r_range[0], r_range[1], n_r_lines)
    theta_starts = np.linspace(0, 2*np.pi, n_theta_lines, endpoint=False)
    
    # Time span for integration
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, resolution)
    
    # Define RHS for integration
    def rhs(t, y):
        r, theta = y
        if r < 0.01:  # Prevent issues at origin
            return [0, 0]
        dr = system.dr_dt(np.array([r]), np.array([theta]))[0]
        dtheta = system.dtheta_dt(np.array([r]), np.array([theta]))[0]
        return [dr, dtheta]
    
    # Integrate and plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, n_r_lines * n_theta_lines))
    color_idx = 0
    
    for r0 in r_starts:
        for theta0 in theta_starts:
            try:
                sol = solve_ivp(rhs, t_span, [r0, theta0], t_eval=t_eval, method='RK45')
                if sol.success:
                    r_traj = sol.y[0]
                    theta_traj = sol.y[1]
                    
                    # Plot trajectory (theta is angular, r is radial in polar plot)
                    ax.plot(theta_traj, r_traj, '-', color=colors[color_idx], 
                            linewidth=0.8, alpha=0.7)
            except:
                pass
            color_idx += 1
    
    # Plot wall as a circle
    if show_wall:
        wall_theta = np.linspace(0, 2*np.pi, 100)
        wall_r = np.full_like(wall_theta, R_w)
        ax.plot(wall_theta, wall_r, 'b-', linewidth=3, 
                label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.1f})')
        
        # Shade wall region
        if system.confinement == 'inside':
            ax.fill_between(wall_theta, R_w, r_range[1], alpha=0.15, color='blue')
        else:
            ax.fill_between(wall_theta, 0, R_w, alpha=0.15, color='blue')
    
    # Mark fixed points
    if show_fixed_points:
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                marker = 'o'
                color = 'green'
                size = 150
            elif result['stability'] == 'Unstable' and 'Saddle' in result['classification']:
                marker = 'X'
                color = 'red'
                size = 150
            else:
                marker = 's'
                color = 'orange'
                size = 120
            
            # In polar plot: theta is angular, r is radial
            ax.scatter(theta_fp, r_fp, marker=marker, s=80, c=color,
                      edgecolors='black', linewidths=1.5, zorder=10, alpha=0.6,
                      label=f"{result['classification']}")
    
    # Configure polar plot
    ax.set_rlim(r_range)
    ax.set_rticks(np.linspace(r_range[0], r_range[1], 5))
    
    # theta labels (orientation angle)
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                        r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
    
    # Labels
    ax.set_xlabel(r'$\theta$ (orientation relative to radial)', fontsize=11, labelpad=15)
    # Radial label
    ax.text(np.pi/4, r_range[1] * 1.15, r'$\tilde{r}$', fontsize=12, ha='center')
    
    if title is None:
        zoom_str = " [near-wall zoom]" if near_wall_zoom else ""
        title = (f'Polar Phase Portrait ({system.confinement.upper()} confinement){zoom_str}\n'
                f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, '
                f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f}, '
                f'$\\tilde{{k}}$ = {system.k_tilde:.1f}')
    ax.set_title(title, fontsize=12, pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_polar_phase_portrait_quiver(system: dcABPCircularWall,
                                      r_range: Optional[Tuple[float, float]] = None,
                                      resolution: int = 15,
                                      figsize: Tuple[int, int] = (10, 10),
                                      show_fixed_points: bool = True,
                                      show_wall: bool = True,
                                      near_wall_zoom: bool = False,
                                      title: Optional[str] = None) -> plt.Figure:
    R_w = system.R_w_tilde
    
    # Set default r_range based on confinement type
    if r_range is None:
        if near_wall_zoom:
            penetration = 2.0 / system.k_tilde + 0.5
            if system.confinement == 'inside':
                r_range = (max(0.1, R_w - 0.5), R_w + penetration)
            else:
                r_range = (max(0.1, R_w - penetration), R_w + 0.5)
        else:
            if system.confinement == 'inside':
                r_range = (max(0.1, R_w - 1.0), R_w + 2.0)
            else:
                r_range = (max(0.1, R_w - 2.0), R_w + 1.0)
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Create meshgrid in (theta, r) for polar plot
    theta_vals = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    r_vals = np.linspace(r_range[0], r_range[1], resolution)
    THETA, R = np.meshgrid(theta_vals, r_vals)
    
    # Compute vector field
    dR = system.dr_dt(R, THETA)
    dTHETA = system.dtheta_dt(R, THETA)
    
    
    # Normalize for visualization (arrows show direction, color shows magnitude)
    speed = np.sqrt(dR**2 + (R * dTHETA)**2)  # Actual speed in polar coords
    
    # Scale arrows by speed for visibility
    scale_factor = 0.15 * (r_range[1] - r_range[0])
    dR_norm = dR / (speed + 1e-10) * scale_factor
    dTHETA_norm = dTHETA / (speed + 1e-10) * scale_factor
    
    q = ax.quiver(THETA, R, dTHETA_norm, dR_norm, speed,
                  cmap='viridis', alpha=0.8, scale=1, scale_units='xy')
    
    # Colorbar
    cbar = fig.colorbar(q, ax=ax, label='|velocity|', shrink=0.8, pad=0.1)
    
    # Plot wall as a circle
    if show_wall:
        wall_theta = np.linspace(0, 2*np.pi, 100)
        wall_r = np.full_like(wall_theta, R_w)
        ax.plot(wall_theta, wall_r, 'b-', linewidth=3, 
                label=f'Wall ($\\tilde{{R}}_w$ = {R_w:.1f})')
        
        # Shade wall region
        if system.confinement == 'inside':
            ax.fill_between(wall_theta, R_w, r_range[1], alpha=0.15, color='blue')
        else:
            ax.fill_between(wall_theta, 0, R_w, alpha=0.15, color='blue')
    
    # Mark fixed points
    if show_fixed_points:
        fixed_pts = find_fixed_points(system)
        results = stability_analysis(system, verbose=False)
        
        for (r_fp, theta_fp), result in zip(fixed_pts, results):
            if result['stability'] == 'Stable':
                marker = 'o'
                color = 'green'
                size = 150
            elif result['stability'] == 'Unstable' and 'Saddle' in result['classification']:
                marker = 'X'
                color = 'red'
                size = 150
            else:
                marker = 's'
                color = 'orange'
                size = 120
            
            ax.scatter(theta_fp, r_fp, marker=marker, s=80, c=color,
                      edgecolors='black', linewidths=1.5, zorder=10, alpha=0.6,
                      label=f"{result['classification']}")
    
    # Configure polar plot
    ax.set_rlim(r_range)
    ax.set_rticks(np.linspace(r_range[0], r_range[1], 5))
    
    # theta labels
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                        r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
    
    # Labels
    ax.text(np.pi/4, r_range[1] * 1.15, r'$\tilde{r}$', fontsize=12, ha='center')
    
    if title is None:
        zoom_str = " [near-wall zoom]" if near_wall_zoom else ""
        title = (f'Polar Phase Portrait - Quiver ({system.confinement.upper()} confinement){zoom_str}\n'
                f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, '
                f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f}, '
                f'$\\tilde{{k}}$ = {system.k_tilde:.1f}')
    ax.set_title(title, fontsize=12, pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_cartesian_vs_polar_phase_portrait(system: dcABPCircularWall,
                                               figsize: Tuple[int, int] = (16, 7),
                                               near_wall_zoom: bool = False) -> plt.Figure:
    R_w = system.R_w_tilde
    
    # Set r_range
    if system.confinement == 'inside':
        r_range = (max(0.5, R_w - 1), R_w + 2)
    else:
        r_range = (max(0.5, R_w - 2), R_w + 1)
    
    if near_wall_zoom:
        penetration = 2.0 / system.k_tilde + 0.5
        if system.confinement == 'inside':
            r_range_polar = (max(0.1, R_w - 0.5), R_w + penetration)
        else:
            r_range_polar = (max(0.1, R_w - penetration), R_w + 0.5)
    else:
        r_range_polar = r_range
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Left: Cartesian (r, theta) phase portrait
    ax1 = fig.add_subplot(121)
    
    resolution = 25
    r = np.linspace(r_range[0], r_range[1], resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    R, THETA = np.meshgrid(r, theta)
    dR, dTHETA = system.vector_field(R, THETA)
    speed = np.sqrt(dR**2 + dTHETA**2)
    
    strm = ax1.streamplot(R, THETA, dR, dTHETA, color=speed, cmap='viridis',
                          density=1.5, linewidth=0.8, arrowsize=1.2)
    fig.colorbar(strm.lines, ax=ax1, label='|velocity|')
    
    ax1.axvline(x=R_w, color='blue', linewidth=2, linestyle='-', label=f'Wall')
    if system.confinement == 'inside':
        ax1.axvspan(R_w, r_range[1], alpha=0.15, color='blue')
    else:
        ax1.axvspan(r_range[0], R_w, alpha=0.15, color='blue')
    
    # Fixed points on Cartesian plot
    fixed_pts = find_fixed_points(system)
    results = stability_analysis(system, verbose=False)
    for (r_fp, theta_fp), result in zip(fixed_pts, results):
        color = 'green' if result['stability'] == 'Stable' else 'red'
        marker = 'o' if result['stability'] == 'Stable' else 'x'
        ax1.plot(r_fp, theta_fp, marker, markersize=12, markeredgewidth=3, color=color)
    
    ax1.set_xlabel(r'$\tilde{r}$', fontsize=12)
    ax1.set_ylabel(r'$\theta$', fontsize=12)
    ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax1.set_title('Cartesian Phase Portrait\n' + r'($\tilde{r}$ on x-axis, $\theta$ on y-axis)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Right: Polar phase portrait
    ax2 = fig.add_subplot(122, projection='polar')
    
    # Use quiver for cleaner visualization
    theta_vals = np.linspace(0, 2*np.pi, 15, endpoint=False)
    r_vals = np.linspace(r_range_polar[0], r_range_polar[1], 12)
    THETA_p, R_p = np.meshgrid(theta_vals, r_vals)
    
    dR_p = system.dr_dt(R_p, THETA_p)
    dTHETA_p = system.dtheta_dt(R_p, THETA_p)
    
    speed_p = np.sqrt(dR_p**2 + (R_p * dTHETA_p)**2)
    scale_factor = 0.15 * (r_range_polar[1] - r_range_polar[0])
    dR_norm = dR_p / (speed_p + 1e-10) * scale_factor
    dTHETA_norm = dTHETA_p / (speed_p + 1e-10) * scale_factor
    
    q = ax2.quiver(THETA_p, R_p, dTHETA_norm, dR_norm, speed_p,
                   cmap='viridis', alpha=0.8, scale=1, scale_units='xy')
    fig.colorbar(q, ax=ax2, label='|velocity|', shrink=0.8, pad=0.1)
    
    # Wall on polar plot
    wall_theta = np.linspace(0, 2*np.pi, 100)
    wall_r = np.full_like(wall_theta, R_w)
    ax2.plot(wall_theta, wall_r, 'b-', linewidth=3, label='Wall')
    if system.confinement == 'inside':
        ax2.fill_between(wall_theta, R_w, r_range_polar[1], alpha=0.15, color='blue')
    else:
        ax2.fill_between(wall_theta, 0, R_w, alpha=0.15, color='blue')
    
    # Fixed points on polar plot
    for (r_fp, theta_fp), result in zip(fixed_pts, results):
        color = 'green' if result['stability'] == 'Stable' else 'red'
        marker = 'o' if result['stability'] == 'Stable' else 'X'
        ax2.scatter(theta_fp, r_fp, marker=marker, s=150, c=color,
                   edgecolors='black', linewidths=1.5, zorder=10)
    
    ax2.set_rlim(r_range_polar)
    ax2.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax2.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                         r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
    zoom_str = " [zoomed]" if near_wall_zoom else ""
    ax2.set_title(f'Polar Phase Portrait{zoom_str}\n' + r'($\theta$ angular, $\tilde{r}$ radial)', fontsize=11, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
    
    # Overall title
    fig.suptitle(f'{system.confinement.upper()} confinement: '
                f'$\\tilde{{\\omega}}$ = {system.omega_tilde:.2f}, '
                f'$\\tilde{{R}}_w$ = {system.R_w_tilde:.1f}, '
                f'$\\tilde{{k}}$ = {system.k_tilde:.1f}', 
                fontsize=13, y=1.02)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 11: THEORETICAL ANALYSIS DOCUMENTATION
# =============================================================================

def print_theoretical_summary():
    summary = """
================================================================================
THEORETICAL SUMMARY: dcABP CIRCULAR WALL ANALYSIS
================================================================================

COORDINATE SYSTEM:
------------------
- r: radial distance from origin
- phi: polar angle of position
- theta: orientation relative to radial direction (theta=0 points outward)
- Swimming direction: n_hat = (cos(theta+phi), sin(theta+phi))

EQUATIONS OF MOTION (dimensionless):
------------------------------------
dr~/dt~ = cos(theta) + F~_r(r~)
dtheta/dt~ = 1 + omega~ + F~_r(r~)cos(theta) - sin(theta)/r~    <-- curvature term
dphi/dt~ = sin(theta)/r~                                        <-- decouples

FIXED POINT SOLUTION (hard wall, r* = R~_w):
--------------------------------------------
sin(theta*) = 1/(2*R~_w) +/- sqrt(1/(4*R~_w^2) - omega~)

EXISTENCE CONDITION:
--------------------
omega~ <= 1/(4*R~_w^2)

CONFINEMENT CASES:
------------------
INSIDE (particle within circle):
  - Wall pushes INWARD (F_r < 0)
  - Fixed point: cos(theta*) > 0 (particle points outward)
  - theta* = arcsin(s)
  - r* > R_w

OUTSIDE (particle outside circle):
  - Wall pushes OUTWARD (F_r > 0)
  - Fixed point: cos(theta*) < 0 (particle points inward)
  - theta* = pi - arcsin(s)
  - r* < R_w

FLAT WALL LIMIT (R~_w -> infinity):
-----------------------------------
sin(theta*) -> +/- sqrt(-omega~)
(Recovers flat wall result exactly)

================================================================================
"""
    print(summary)


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("dcABP Circular Wall Analysis - Module loaded successfully")
    print("="*70)
    print("\nThis module provides functions for dcABP circular wall analysis.")
    print("Import it in your playground file to use the functions.")
    print("\nMain functions available:")
    print("  - dcABPCircularWall: System class")
    print("  - find_fixed_points: Find fixed points analytically")
    print("  - stability_analysis: Full stability analysis")
    print("  - plot_phase_portrait: Cartesian phase portrait")
    print("  - plot_polar_phase_portrait: Polar (bullseye) phase portrait")
    print("  - plot_polar_phase_portrait_quiver: Polar quiver plot")
    print("  - compare_cartesian_vs_polar_phase_portrait: Side-by-side comparison")
    print("  - parameter_sweep_omega/R_w/k: Parameter sweeps")
    print("  - bifurcation_diagram_omega/R_w: Bifurcation diagrams")
    print("  - integrate_trajectory: Trajectory integration")
    print("  - print_theoretical_summary: Print theory summary")


# =============================================================================
# SECTION 12: DETAILED FIXED POINT ANALYSIS WITH BRANCH TRACKING
# =============================================================================

def find_fixed_points_detailed(system, verbose=False):
    omega = system.omega_tilde
    R_w = system.R_w_tilde
    k = system.k_tilde
    results = []
    
    discriminant = 1/(4 * R_w**2) - omega
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DETAILED FIXED POINT ANALYSIS - CIRCULAR WALL")
        print(f"{'='*60}")
        print(f"Parameters: omega~ = {omega:.4f}, R~_w = {R_w:.4f}, k~ = {k:.4f}")
        print(f"Confinement: {system.confinement.upper()}")
        print(f"Discriminant = {discriminant:.6f}")
    
    if discriminant < 0:
        if verbose:
            print("Discriminant < 0: NO FIXED POINTS")
        return results
    
    sqrt_disc = np.sqrt(discriminant)
    s_plus = 1/(2*R_w) + sqrt_disc
    s_minus = 1/(2*R_w) - sqrt_disc
    
    if verbose:
        print(f"s_plus = {s_plus:.6f}, s_minus = {s_minus:.6f}")
    
    for branch_name, s_value in [('plus', s_plus), ('minus', s_minus)]:
        result = {'branch': branch_name, 'branch_value': s_value, 'valid': False,
                  'sin_theta_star': None, 'cos_theta_star': None, 
                  'theta_star': None, 'r_star': None, 'validity_reason': None}
        
        if abs(s_value) > 1:
            result['validity_reason'] = f'|s| = {abs(s_value):.4f} > 1'
            results.append(result)
            continue
        
        arcsin_s = np.arcsin(s_value)
        theta_candidates = [arcsin_s, np.pi - arcsin_s] if s_value >= 0 else [2*np.pi + arcsin_s, np.pi - arcsin_s]
        
        valid_theta = None
        for theta_cand in theta_candidates:
            cos_theta = np.cos(theta_cand)
            if system.confinement == 'inside' and cos_theta > 0:
                valid_theta = theta_cand
                break
            elif system.confinement == 'outside' and cos_theta < 0:
                valid_theta = theta_cand
                break
        
        if valid_theta is None:
            result['validity_reason'] = 'No valid theta'
            results.append(result)
            continue
        
        result['theta_star'] = valid_theta
        result['sin_theta_star'] = np.sin(valid_theta)
        result['cos_theta_star'] = np.cos(valid_theta)
        result['r_star'] = R_w + np.cos(valid_theta) / k
        result['valid'] = True
        
        if verbose:
            print(f"  {branch_name}: sin={result['sin_theta_star']:.4f}, theta*={np.degrees(valid_theta):.1f} deg")
        
        results.append(result)
    
    return results


def full_stability_analysis(system, verbose=True):
    fp_details = find_fixed_points_detailed(system, verbose=verbose)
    results = []
    
    for fp in fp_details:
        if not fp['valid']:
            results.append(fp)
            continue
        
        J = compute_jacobian(system, fp['r_star'], fp['theta_star'])
        stability = classify_fixed_point(J)
        fp.update(stability)
        fp['jacobian'] = J
        
        if verbose:
            print(f"  {fp['branch']} branch: {stability['classification']} ({stability['stability']})")
        
        results.append(fp)
    
    return results


# =============================================================================
# SECTION 13: BASIN OF ATTRACTION
# =============================================================================

def classify_trajectory_fate(system, initial_condition, t_max=200, tolerance=0.1):
    t, r, theta, phi = integrate_full_trajectory(system, (initial_condition[0], initial_condition[1], 0.0), (0, t_max), 2000)
    
    result = {'initial': initial_condition, 'final_position': (r[-1], theta[-1]),
              'fate': 'unknown', 'touches_wall': False}
    
    if system.confinement == 'inside':
        result['touches_wall'] = np.any(r > system.R_w_tilde)
    else:
        result['touches_wall'] = np.any(r < system.R_w_tilde)
    
    fp_results = full_stability_analysis(system, verbose=False)
    stable_fps = [(fp['r_star'], fp['theta_star']) for fp in fp_results if fp['valid'] and fp.get('stability') == 'Stable']
    
    final_r, final_theta = r[-1], theta[-1] % (2*np.pi)
    
    for i, (r_fp, theta_fp) in enumerate(stable_fps):
        theta_fp_mod = theta_fp % (2*np.pi)
        if abs(final_r - r_fp) < tolerance and min(abs(final_theta - theta_fp_mod), 2*np.pi - abs(final_theta - theta_fp_mod)) < tolerance:
            result['fate'] = 'stable_fp'
            return result
    
    if len(r) > 100:
        if np.max(r[-500:]) - np.min(r[-500:]) > 0.1:
            result['fate'] = 'orbits'
    
    return result


def compute_basin_of_attraction(system, r_range=None, theta_range=(0, 2*np.pi), n_r=30, n_theta=30, t_max=200, verbose=True):
    R_w = system.R_w_tilde
    if r_range is None:
        r_range = (max(0.5, R_w - 2), R_w + 1.5) if system.confinement == 'inside' else (max(0.5, R_w - 1.5), R_w + 2)
    
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta, endpoint=False)
    
    fate_grid = np.empty((n_r, n_theta), dtype=object)
    touches_wall_grid = np.zeros((n_r, n_theta), dtype=bool)
    
    total = n_r * n_theta
    if verbose:
        print(f"Computing basin: {total} trajectories...")
    
    for i, r0 in enumerate(r_vals):
        for j, theta0 in enumerate(theta_vals):
            result = classify_trajectory_fate(system, (r0, theta0), t_max)
            fate_grid[i, j] = result['fate']
            touches_wall_grid[i, j] = result['touches_wall']
    
    fate_counts = {fate: np.sum(fate_grid == fate) for fate in np.unique(fate_grid)}
    wall_touch_count = np.sum(touches_wall_grid)
    wall_touch_to_stable = np.sum((touches_wall_grid) & (fate_grid == 'stable_fp'))
    
    if verbose:
        print(f"Results: {fate_counts}")
    
    return {'r_vals': r_vals, 'theta_vals': theta_vals, 'fate_grid': fate_grid,
            'touches_wall_grid': touches_wall_grid, 'fate_counts': fate_counts,
            'wall_touch_count': wall_touch_count, 'wall_touch_to_stable': wall_touch_to_stable, 'system': system}


def plot_basin_of_attraction(basin_data, figsize=(14, 6)):
    from matplotlib.patches import Patch
    r_vals, theta_vals = basin_data['r_vals'], basin_data['theta_vals']
    fate_grid, touches_wall_grid = basin_data['fate_grid'], basin_data['touches_wall_grid']
    system = basin_data['system']
    
    R, THETA = np.meshgrid(r_vals, theta_vals, indexing='ij')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    fate_map = {'stable_fp': 0, 'near_saddle': 1, 'escapes_inward': 2, 'escapes_outward': 2, 'orbits': 3, 'unknown': 4}
    fate_numeric = np.array([[fate_map.get(fate_grid[i,j], 4) for j in range(len(theta_vals))] for i in range(len(r_vals))])
    
    cmap = plt.cm.colors.ListedColormap(['green', 'orange', 'blue', 'purple', 'gray'])
    ax1.pcolormesh(R, THETA, fate_numeric, cmap=cmap, vmin=-0.5, vmax=4.5)
    ax1.axvline(x=system.R_w_tilde, color='black', linewidth=2)
    ax1.set_xlabel(r'$\tilde{r}$')
    ax1.set_ylabel(r'$\theta$')
    ax1.set_title('Basin of Attraction')
    ax1.legend(handles=[Patch(facecolor='green', label='Stable'), Patch(facecolor='purple', label='Orbits')])
    
    wall_fate = np.array([[0 if not touches_wall_grid[i,j] else (1 if fate_grid[i,j]=='stable_fp' else 2) 
                           for j in range(len(theta_vals))] for i in range(len(r_vals))])
    cmap2 = plt.cm.colors.ListedColormap(['lightgray', 'green', 'red'])
    ax2.pcolormesh(R, THETA, wall_fate, cmap=cmap2, vmin=-0.5, vmax=2.5)
    ax2.axvline(x=system.R_w_tilde, color='black', linewidth=2)
    ax2.set_xlabel(r'$\tilde{r}$')
    ax2.set_ylabel(r'$\theta$')
    ax2.set_title('Wall-Touching Trajectories')
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 14: sin(theta*) ANALYSIS
# =============================================================================

def analyze_sin_theta_branches(R_w_range=(0.5, 20.0), omega_range=(-0.99, 0.01), n_Rw=50, n_omega=50, confinement='inside', k_tilde=10.0):
    R_w_vals = np.linspace(R_w_range[0], R_w_range[1], n_Rw)
    omega_vals = np.linspace(omega_range[0], omega_range[1], n_omega)
    
    plus_sin = np.full((n_omega, n_Rw), np.nan)
    plus_stability = np.full((n_omega, n_Rw), np.nan)
    minus_sin = np.full((n_omega, n_Rw), np.nan)
    minus_stability = np.full((n_omega, n_Rw), np.nan)
    n_fixed_points = np.zeros((n_omega, n_Rw))
    
    for i, omega in enumerate(omega_vals):
        for j, R_w in enumerate(R_w_vals):
            system = dcABPCircularWall(omega, R_w, k_tilde, confinement)
            results = full_stability_analysis(system, verbose=False)
            
            for fp in results:
                if fp['valid']:
                    n_fixed_points[i, j] += 1
                    stab = 1 if fp.get('stability') == 'Stable' else 0
                    if fp['branch'] == 'plus':
                        plus_sin[i, j] = fp['sin_theta_star']
                        plus_stability[i, j] = stab
                    else:
                        minus_sin[i, j] = fp['sin_theta_star']
                        minus_stability[i, j] = stab
    
    return {'R_w_vals': R_w_vals, 'omega_vals': omega_vals, 'plus_sin': plus_sin,
            'plus_stability': plus_stability, 'minus_sin': minus_sin,
            'minus_stability': minus_stability, 'n_fixed_points': n_fixed_points, 'confinement': confinement}


def plot_sin_theta_analysis(analysis_data, figsize=(16, 12)):
    R_w = analysis_data['R_w_vals']
    omega = analysis_data['omega_vals']
    OMEGA, RW = np.meshgrid(omega, R_w, indexing='ij')
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    axes[0,0].pcolormesh(OMEGA, RW, analysis_data['n_fixed_points'], cmap='viridis', vmin=0, vmax=2)
    axes[0,0].set_title('# Fixed Points')
    
    axes[0,1].pcolormesh(OMEGA, RW, analysis_data['plus_sin'], cmap='RdBu', vmin=-1, vmax=1)
    axes[0,1].set_title("sin(theta*) '+' branch")
    
    axes[0,2].pcolormesh(OMEGA, RW, analysis_data['minus_sin'], cmap='RdBu', vmin=-1, vmax=1)
    axes[0,2].set_title("sin(theta*) '-' branch")
    
    axes[1,0].pcolormesh(OMEGA, RW, analysis_data['plus_stability'], cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,0].set_title("'+' stability")
    
    axes[1,1].pcolormesh(OMEGA, RW, analysis_data['minus_stability'], cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,1].set_title("'-' stability")
    
    for ax in axes.flatten()[:5]:
        ax.set_xlabel(r'$\tilde{\omega}$')
        ax.set_ylabel(r'$\tilde{R}_w$')
    
    axes[1,2].axis('off')
    fig.suptitle(f"sin(theta*) Analysis ({analysis_data['confinement'].upper()})", fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 15: TWO-PARAMETER PHASE DIAGRAM
# =============================================================================

def compute_phase_diagram(omega_range=(-1.5, 0.5), inv_Rw_range=(0.0, 2.0), n_omega=100, n_inv_Rw=100, confinement='inside', k_tilde=10.0):
    omega_vals = np.linspace(omega_range[0], omega_range[1], n_omega)
    inv_Rw_vals = np.linspace(inv_Rw_range[0], inv_Rw_range[1], n_inv_Rw)
    
    n_fixed_points = np.zeros((n_omega, n_inv_Rw))
    stable_sin_sign = np.full((n_omega, n_inv_Rw), np.nan)
    saddle_sin_sign = np.full((n_omega, n_inv_Rw), np.nan)
    
    for i, omega in enumerate(omega_vals):
        for j, inv_Rw in enumerate(inv_Rw_vals):
            if inv_Rw < 1e-6:
                if -1 < omega < 0:
                    n_fixed_points[i, j] = 2
                    stable_sin_sign[i, j] = 1
                    saddle_sin_sign[i, j] = -1
                continue
            
            R_w = 1.0 / inv_Rw
            system = dcABPCircularWall(omega, R_w, k_tilde, confinement)
            results = full_stability_analysis(system, verbose=False)
            
            for fp in results:
                if fp['valid']:
                    n_fixed_points[i, j] += 1
                    if fp.get('stability') == 'Stable':
                        stable_sin_sign[i, j] = np.sign(fp['sin_theta_star'])
                    else:
                        saddle_sin_sign[i, j] = np.sign(fp['sin_theta_star'])
    
    return {'omega_vals': omega_vals, 'inv_Rw_vals': inv_Rw_vals, 'n_fixed_points': n_fixed_points,
            'stable_sin_sign': stable_sin_sign, 'saddle_sin_sign': saddle_sin_sign, 'confinement': confinement}


def plot_phase_diagram(phase_data, figsize=(14, 10)):
    omega = phase_data['omega_vals']
    inv_Rw = phase_data['inv_Rw_vals']
    OMEGA, INV_RW = np.meshgrid(omega, inv_Rw, indexing='ij')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_discrete = ListedColormap(['#440154', '#21918c', '#fde725'])
    norm_discrete = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_discrete.N)
    c = axes[0,0].pcolormesh(OMEGA, INV_RW, phase_data['n_fixed_points'], cmap=cmap_discrete, norm=norm_discrete)
    inv_Rw_fine = np.linspace(0.01, inv_Rw[-1], 200)
    axes[0,0].plot(inv_Rw_fine**2/4, inv_Rw_fine, 'r-', lw=2, label='Boundary')
    axes[0,0].set_title('# Fixed Points')
    axes[0,0].legend()
    cbar = fig.colorbar(c, ax=axes[0,0], ticks=[0, 1, 2])
    
    axes[0,1].pcolormesh(OMEGA, INV_RW, phase_data['n_fixed_points'] > 0, cmap='RdYlGn')
    axes[0,1].set_title('Has Fixed Points')
    
    c = axes[1,0].pcolormesh(OMEGA, INV_RW, phase_data['stable_sin_sign'], cmap='RdBu', vmin=-1, vmax=1)
    axes[1,0].set_title('sin(theta*) at Stable FP')
    fig.colorbar(c, ax=axes[1,0])
    
    c = axes[1,1].pcolormesh(OMEGA, INV_RW, phase_data['saddle_sin_sign'], cmap='RdBu', vmin=-1, vmax=1)
    axes[1,1].set_title('sin(theta*) at Saddle FP')
    fig.colorbar(c, ax=axes[1,1])
    
    for ax in axes.flatten():
        ax.set_xlabel(r'$\tilde{\omega}$')
        ax.set_ylabel(r'$1/\tilde{R}_w$')
    
    fig.suptitle(f"Phase Diagram ({phase_data['confinement'].upper()})", fontsize=14)
    plt.tight_layout()
    return fig
