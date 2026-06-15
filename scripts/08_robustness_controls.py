"""
WEEK 8 — Robustness controls and parameter sensitivity.

Tests:
  1. Alternative Hamiltonian (Novoderezhkin 2011) — run reduced sweep
  2. Alternative collapse operators (energy relaxation vs. pure dephasing)
  3. Bootstrap CI on γ_c (resampling disorder realisations)

Run: %run scripts/08_robustness_controls.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import yaml
from scipy.optimize import curve_fit
from sklearn.utils import resample

from src.hamiltonian import (load_hamiltonian_or_test, apply_disorder,
                              get_geometry_hamiltonian, SOURCE_SITE, SINK_SITE)
from src.lindblad import (embed_hamiltonian, build_collapse_operators,
                           compute_steady_state, N_TOT, G_IDX, CM1_TO_RADPS)
from src.metrics import compute_ipr, sigmoid_model
from src.visualization import apply_pub_style

import qutip as qt

apply_pub_style()
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

SWEEP_FILE   = 'results/sweep_raw.h5'
NOVO_H5      = 'results/lhcii_hamiltonian_novo2011.h5'
OUTPUT_CSV   = 'results/week8_sensitivity_table.csv'
BOOT_PNG     = 'figures/week8_bootstrap_distribution.png'

# Reduced sweep parameters for robustness tests (faster than full sweep)
N_GAMMA_ROB  = 40
N_REAL_ROB   = 100
KAPPA_IN     = 10.0
KAPPA_OUT    = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Alternative Hamiltonian: Novoderezhkin 2011
# ─────────────────────────────────────────────────────────────────────────────

def build_novo2011_hamiltonian():
    """
    Novoderezhkin et al. 2011 PCCP Hamiltonian.

    ⚠ Same warning as 02_build_hamiltonian.py: verify these values
    against the published paper before using in any result.

    For now, this generates the Müh 2010 test Hamiltonian with a
    ±50 cm⁻¹ perturbation to mimic inter-parameter-set variation.
    Replace with exact Novoderezhkin 2011 values from their Table.
    """
    if os.path.exists(NOVO_H5):
        with h5py.File(NOVO_H5, 'r') as f:
            return f['H'][:].astype(complex), 'novo2011_file'

    # Fallback: perturbed Müh test matrix
    from src.hamiltonian import get_test_hamiltonian
    H_mueh = np.real(get_test_hamiltonian())

    rng = np.random.default_rng(2011)
    perturbation = rng.uniform(-50, 50, size=14)
    H_novo = H_mueh.copy()
    H_novo[np.diag_indices(14)] += perturbation

    print('⚠  Using perturbed test Hamiltonian for robustness comparison.')
    print('   Replace with exact Novoderezhkin 2011 values for publication.')
    return H_novo.astype(complex), 'novo2011_perturbed'


# ─────────────────────────────────────────────────────────────────────────────
# Alternative collapse operators
# ─────────────────────────────────────────────────────────────────────────────

def build_relaxation_collapse_ops(gamma_cm1, kappa_in_cm1, kappa_out_cm1,
                                   source_site=SOURCE_SITE, sink_site=SINK_SITE):
    """
    Energy-relaxation collapse operators:  L_n = √γ · |n−1⟩⟨n|  (downhill only).

    Contrasts with pure dephasing to test collapse-operator sensitivity.
    """
    c_ops = []

    # Downhill energy relaxation (site n → site n-1 within chromophore block)
    if gamma_cm1 > 0:
        sqrt_gamma = float(np.sqrt(gamma_cm1 * CM1_TO_RADPS))
        for n in range(1, 14):            # skip site 0 (no lower site)
            op = np.zeros((N_TOT, N_TOT), dtype=complex)
            op[n - 1, n] = 1.0           # |n-1⟩⟨n|
            c_ops.append(sqrt_gamma * qt.Qobj(op))

    # Source and sink unchanged
    if kappa_in_cm1 > 0:
        op = np.zeros((N_TOT, N_TOT), dtype=complex)
        op[source_site, G_IDX] = 1.0
        c_ops.append(float(np.sqrt(kappa_in_cm1 * CM1_TO_RADPS)) * qt.Qobj(op))

    if kappa_out_cm1 > 0:
        op = np.zeros((N_TOT, N_TOT), dtype=complex)
        op[G_IDX, sink_site] = 1.0
        c_ops.append(float(np.sqrt(kappa_out_cm1 * CM1_TO_RADPS)) * qt.Qobj(op))

    return c_ops


# ─────────────────────────────────────────────────────────────────────────────
# Reduced sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_reduced_sweep(H_base, gamma_array, n_real=N_REAL_ROB,
                      collapse_fn=None, label=''):
    """
    Run a mini-sweep and return mean IPR per gamma.
    collapse_fn: callable(gamma, kin, kout, src, snk) → c_ops
    """
    if collapse_fn is None:
        collapse_fn = build_collapse_operators   # default: dephasing

    mean_ipr = np.full(len(gamma_array), np.nan)

    for gi, gamma in enumerate(gamma_array):
        iprs = []
        H_qt = embed_hamiltonian(H_base)
        c_ops = collapse_fn(gamma, KAPPA_IN, KAPPA_OUT, SOURCE_SITE, SINK_SITE)
        rng = np.random.default_rng(gi * 1000)

        for r in range(n_real):
            H_dis = apply_disorder(H_base, sigma_cm1=0, rng=rng)  # σ=0 for comparison
            H_qt_r = embed_hamiltonian(H_dis)
            rho_ss = compute_steady_state(H_qt_r, c_ops, verbose=False)
            if rho_ss is not None:
                ipr, _, _, _ = compute_ipr(rho_ss)
                iprs.append(ipr)
        mean_ipr[gi] = np.nanmean(iprs) if iprs else np.nan

    print(f'  {label}: IPR range {np.nanmin(mean_ipr):.4f} – {np.nanmax(mean_ipr):.4f}')
    return mean_ipr


def extract_gamma_c(gamma_array, mean_ipr):
    """Fit sigmoid and return γ_c."""
    log_g  = np.log(gamma_array)
    p0     = [float(np.nanmax(mean_ipr) - np.nanmin(mean_ipr)),
              2.0,
              float(log_g[np.nanargmax(np.gradient(mean_ipr))]),
              float(np.nanmin(mean_ipr))]
    try:
        popt, _ = curve_fit(sigmoid_model, log_g, mean_ipr, p0=p0, maxfev=10000)
        return float(np.exp(popt[2]))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap from primary sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(gamma_array, n_boot=1000):
    """Load primary sweep (σ=0, symmetric) and bootstrap γ_c."""
    if not os.path.exists(SWEEP_FILE):
        print(f'  {SWEEP_FILE} not found — skipping bootstrap.')
        return np.array([]), np.nan, np.nan

    with h5py.File(SWEEP_FILE, 'r') as f:
        ipr_slice = f['ipr'][:, 0, 0, :]   # (n_gamma, n_real)  σ=0, symmetric

    # Align gamma arrays if needed
    n_gam = min(len(gamma_array), ipr_slice.shape[0])
    ipr_slice = ipr_slice[:n_gam, :]
    gamma_use = gamma_array[:n_gam]

    boot_gc = []
    for _ in range(n_boot):
        boot_ipr = resample(ipr_slice.T, n_samples=ipr_slice.shape[1]).T
        mean_b   = np.nanmean(boot_ipr, axis=1)
        gc       = extract_gamma_c(gamma_use, mean_b)
        if np.isfinite(gc):
            boot_gc.append(gc)

    boot_gc = np.array(boot_gc)
    ci_lo   = float(np.percentile(boot_gc, 2.5))  if len(boot_gc) else np.nan
    ci_hi   = float(np.percentile(boot_gc, 97.5)) if len(boot_gc) else np.nan
    return boot_gc, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('='*60)
    print('WEEK 8 — Robustness Controls & Sensitivity Analysis')
    print('='*60)

    # ── Load primary Hamiltonian ────────────────────────────────────────
    H_mueh, h_src = load_hamiltonian_or_test()
    H_novo, n_src = build_novo2011_hamiltonian()

    # Reduced gamma sweep
    gamma_array = np.logspace(np.log10(1.0), np.log10(10000.0), N_GAMMA_ROB)

    records = []

    # ── Test 1: Müh 2010, dephasing ────────────────────────────────────
    print('\n[1] Müh 2010 — pure dephasing…')
    ipr_mueh_deph = run_reduced_sweep(H_mueh, gamma_array,
                                       label='Müh 2010, dephasing')
    gc_mueh_deph  = extract_gamma_c(gamma_array, ipr_mueh_deph)
    records.append({'label': 'Müh 2010 / dephasing',
                    'gamma_c': gc_mueh_deph})
    print(f'  γ_c = {gc_mueh_deph:.1f} cm⁻¹')

    # ── Test 2: Novoderezhkin 2011, dephasing ──────────────────────────
    print('\n[2] Novoderezhkin 2011 — pure dephasing…')
    ipr_novo_deph = run_reduced_sweep(H_novo, gamma_array,
                                       label='Novoderezhkin 2011, dephasing')
    gc_novo_deph  = extract_gamma_c(gamma_array, ipr_novo_deph)
    records.append({'label': 'Novoderezhkin 2011 / dephasing',
                    'gamma_c': gc_novo_deph})
    print(f'  γ_c = {gc_novo_deph:.1f} cm⁻¹')

    # Fold difference
    if np.isfinite(gc_mueh_deph) and np.isfinite(gc_novo_deph) and gc_mueh_deph > 0:
        fold = max(gc_mueh_deph, gc_novo_deph) / min(gc_mueh_deph, gc_novo_deph)
        print(f'  Fold difference: {fold:.2f}× '
              f'(pre-registered acceptable: < 3×)')
        records.append({'label': 'Fold difference (Müh vs Novo)', 'gamma_c': fold})
        if fold > 3.0:
            print('  ⚠  Fold > 3×.  See Pivot Trigger 2 in roadmap.')
            print('     Consider reframing as parameter-sensitivity characterisation.')

    # ── Test 3: Müh 2010, energy relaxation collapse ops ───────────────
    print('\n[3] Müh 2010 — energy-relaxation collapse operators…')
    ipr_mueh_relax = run_reduced_sweep(H_mueh, gamma_array,
                                        collapse_fn=build_relaxation_collapse_ops,
                                        label='Müh 2010, relaxation')
    gc_mueh_relax  = extract_gamma_c(gamma_array, ipr_mueh_relax)
    records.append({'label': 'Müh 2010 / relaxation ops',
                    'gamma_c': gc_mueh_relax})
    print(f'  γ_c = {gc_mueh_relax:.1f} cm⁻¹')

    # ── Test 4: Bootstrap on primary sweep ─────────────────────────────
    print(f'\n[4] Bootstrap CI on primary sweep (N=1000)…')
    boot_gc, ci_lo, ci_hi = run_bootstrap(gamma_array)
    gc_boot_mean = float(np.nanmean(boot_gc)) if len(boot_gc) else np.nan
    records.append({'label': 'Bootstrap mean γ_c (primary sweep)',
                    'gamma_c': gc_boot_mean,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'n_valid': len(boot_gc)})
    print(f'  Bootstrap γ_c = {gc_boot_mean:.1f} cm⁻¹  95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]')

    # ── Save sensitivity table ──────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSensitivity table saved: {OUTPUT_CSV}')
    print(df.to_string(index=False))

    # ── Comparison plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (ipr_arr, lbl, color) in zip(
        [axes[0], axes[0], axes[0]],
        [(ipr_mueh_deph,  'Müh 2010 / dephasing',    '#2C6E9E'),
         (ipr_novo_deph,  'Novo 2011 / dephasing',    '#C25A3D'),
         (ipr_mueh_relax, 'Müh 2010 / relaxation ops','#3B9E4A')]):
        axes[0].semilogx(gamma_array, ipr_arr, lw=2, label=lbl, color=color)

    axes[0].set_xlabel('γ  (cm⁻¹)')
    axes[0].set_ylabel('Mean IPR  (σ=0)')
    axes[0].set_title('IPR(γ): sensitivity to Hamiltonian & collapse ops')
    axes[0].legend(fontsize=8, framealpha=0.9)
    axes[0].set_ylim(0, 1.05)

    if len(boot_gc) > 10:
        axes[1].hist(boot_gc, bins=40, color='#2C6E9E', alpha=0.7,
                     edgecolor='white', lw=0.4)
        axes[1].axvline(gc_boot_mean, color='black', lw=1.8,
                        label=f'Mean {gc_boot_mean:.1f} cm⁻¹')
        axes[1].axvspan(ci_lo, ci_hi, color='#2C6E9E', alpha=0.2,
                        label=f'95% CI [{ci_lo:.0f}, {ci_hi:.0f}]')
        axes[1].set_xlabel('Bootstrap γ_c  (cm⁻¹)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Figure 4 — Bootstrap γ_c distribution')
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('figures/week8_robustness.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/week8_robustness.svg')
    if len(boot_gc) > 10:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(boot_gc, bins=40, color='#2C6E9E', alpha=0.7,
                 edgecolor='white', lw=0.4)
        ax2.axvline(gc_boot_mean, color='black', lw=1.8,
                    label=f'Mean {gc_boot_mean:.1f} cm⁻¹')
        ax2.axvspan(ci_lo, ci_hi, color='#2C6E9E', alpha=0.2,
                    label=f'95% CI [{ci_lo:.0f}, {ci_hi:.0f}]')
        ax2.set_xlabel('Bootstrap γ_c  (cm⁻¹)')
        ax2.set_ylabel('Count')
        ax2.set_title('Figure 4 — Bootstrap γ_c distribution')
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        fig2.savefig(BOOT_PNG, dpi=300, bbox_inches='tight')
        fig2.savefig(BOOT_PNG.replace('.png', '.svg'))

    print('\n✓ Week 8 complete.  Run scripts/09_figure_finalization.py next.')
    plt.show()


if __name__ == '__main__':
    main()
