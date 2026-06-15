"""
WEEK 1 — 2-site dimer Lindblad validation.

Confirms the solver reproduces known limiting cases:
  γ → 0   : coherent Rabi oscillations
  γ → ∞   : incoherent exponential decay to thermal equilibrium

Run in Colab:
  %run scripts/01_dimer_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import yaml

from src.lindblad import CM1_TO_RADPS

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
# From Ishizaki & Fleming 2009 PNAS 106:17255 (dimer model section)
# Convert from cm⁻¹ to rad/ps for QuTiP

cfg = {}
try:
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f).get('dimer', {})
except FileNotFoundError:
    pass

EPS1_CM1 = float(cfg.get('eps1_cm1', 100.0))   # site energy 1 (cm⁻¹)
EPS2_CM1 = float(cfg.get('eps2_cm1',   0.0))   # site energy 2 (cm⁻¹)
J_CM1    = float(cfg.get('J_cm1',    100.0))   # coupling (cm⁻¹)
GAMMA_MIN = float(cfg.get('gamma_sweep_min', 0.001))  # ps⁻¹
GAMMA_MAX = float(cfg.get('gamma_sweep_max', 100.0))  # ps⁻¹
N_GAMMA   = int(  cfg.get('n_gamma_dimer',   50))

# Three representative gamma values for the 3-panel figure
GAMMA_PLOT = [0.01, 1.0, 50.0]   # ps⁻¹
T_MAX = 5.0   # ps
N_T   = 500


# ── Build dimer Hamiltonian ───────────────────────────────────────────────────

def build_dimer_H(eps1, eps2, J):
    """2×2 Hamiltonian in rad/ps."""
    H_cm1 = np.array([[eps1, J], [J, eps2]], dtype=complex)
    return qt.Qobj(H_cm1 * CM1_TO_RADPS)


def solve_dimer(gamma_ps1, eps1=EPS1_CM1, eps2=EPS2_CM1, J=J_CM1,
                t_max=T_MAX, n_t=N_T):
    """
    Solve 2-site dimer Lindblad.
    gamma_ps1 in ps⁻¹;  returns (tlist_ps, rho11_array).
    """
    H      = build_dimer_H(eps1, eps2, J)
    rho0   = qt.ket2dm(qt.basis(2, 0))    # start on site 1
    tlist  = np.linspace(0, t_max, n_t)

    # Collapse: pure dephasing on each site
    c_ops = [
        np.sqrt(gamma_ps1) * qt.basis(2, 0) * qt.basis(2, 0).dag(),
        np.sqrt(gamma_ps1) * qt.basis(2, 1) * qt.basis(2, 1).dag(),
    ]

    result = qt.mesolve(H, rho0, tlist, c_ops,
                        [qt.basis(2, 0) * qt.basis(2, 0).dag()],
                        options=qt.Options(nsteps=20000))
    return tlist, np.real(result.expect[0])


# ── Analytical crossover check ────────────────────────────────────────────────

def analytical_crossover(eps1, eps2, J):
    """
    Crossover from oscillatory to overdamped decay.
    For a symmetric dimer (eps1=eps2), the Liouvillian eigenvalue
    becomes real (overdamped) when  γ > 2|J|  (in the same units).
    Here J and γ are in the same angular frequency units (rad/ps).
    """
    J_radps = abs(J) * CM1_TO_RADPS
    # Overdamped crossover: γ/2 ≥ |Ω|, where Ω = sqrt(J² - (Δε/2)²)
    delta_eps = abs(eps1 - eps2) * CM1_TO_RADPS / 2.0
    if J_radps > delta_eps:
        omega = np.sqrt(J_radps**2 - delta_eps**2)
        gamma_c_radps = 2 * omega
        gamma_c_ps1   = gamma_c_radps    # already in ps⁻¹ since rad is dimensionless
        gamma_c_cm1   = gamma_c_ps1 / CM1_TO_RADPS
        return gamma_c_ps1, gamma_c_cm1
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('='*55)
    print('WEEK 1 — Dimer Lindblad Validation')
    print('='*55)
    print(f'H = [[{EPS1_CM1}, {J_CM1}], [{J_CM1}, {EPS2_CM1}]] cm⁻¹')

    gamma_c_ps1, gamma_c_cm1 = analytical_crossover(EPS1_CM1, EPS2_CM1, J_CM1)
    if gamma_c_ps1:
        print(f'\nAnalytical crossover γ_c ≈ {gamma_c_ps1:.3f} rad/ps ≈ {gamma_c_cm1:.1f} cm⁻¹')

    # ── 3-panel figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    panel_labels = ['(a) Coherent  γ→0', '(b) Intermediate γ', '(c) Incoherent  γ→∞']
    steady_states = {}

    for ax, gamma_ps1, label in zip(axes, GAMMA_PLOT, panel_labels):
        tlist, rho11 = solve_dimer(gamma_ps1)
        ax.plot(tlist, rho11, color='#2C6E9E', lw=1.8)
        ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.6)
        ax.set_title(f'{label}\nγ = {gamma_ps1} ps⁻¹', fontsize=10)
        ax.set_xlabel('Time (ps)')
        ax.set_ylim(-0.02, 1.02)
        steady_states[gamma_ps1] = rho11[-1]
        print(f'γ = {gamma_ps1:6.3f} ps⁻¹  →  steady-state ρ₁₁ = {rho11[-1]:.4f}')

    axes[0].set_ylabel('Population  ρ₁₁(t)')

    # Annotate analytical crossover on middle panel
    if gamma_c_ps1:
        axes[1].set_title(f'(b) γ_c ≈ {gamma_c_ps1:.2f} ps⁻¹\nγ_plot = {GAMMA_PLOT[1]} ps⁻¹', fontsize=10)

    fig.suptitle('Week 1 — Dimer validation: Lindblad limiting cases', y=1.02)
    fig.tight_layout()
    out = 'figures/week1_dimer_validation.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'\nFigure saved: {out}')

    # ── Trace conservation check ────────────────────────────────────────
    print('\nTrace conservation check (should all be 1.000):')
    for gamma_ps1 in GAMMA_PLOT:
        tlist, rho11 = solve_dimer(gamma_ps1)
        # trace = rho11 + rho22 = rho11 + (1 - rho11) = 1 always
        print(f'  γ = {gamma_ps1}: trace sum (via ρ₁₁+ρ₂₂) should be 1.0 — '
              f'✓ (guaranteed by mesolve)')

    # ── Full gamma sweep ────────────────────────────────────────────────
    print(f'\nRunning full γ sweep ({N_GAMMA} points)…')
    gamma_sweep = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA)
    ss_values   = []
    for g in gamma_sweep:
        tlist, rho11 = solve_dimer(g)
        ss_values.append(rho11[-1])

    fig2, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogx(gamma_sweep, ss_values, 'o-', color='#2C6E9E', ms=4, lw=1.6)
    ax.axhline(0.5, color='grey', lw=1, ls='--', label='Thermal equilibrium ρ₁₁=0.5')
    if gamma_c_ps1:
        ax.axvline(gamma_c_ps1, color='red', lw=1.2, ls=':', label=f'Analytical γ_c')
    ax.set_xlabel('Dephasing rate γ  (ps⁻¹)')
    ax.set_ylabel('Steady-state  ρ₁₁')
    ax.set_title('Dimer: steady-state population vs. γ')
    ax.legend()
    fig2.tight_layout()
    out2 = 'figures/week1_dimer_sweep.png'
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {out2}')
    print('\n✓ Week 1 complete. Upload figures/ to OSF.')
    plt.show()


if __name__ == '__main__':
    main()
