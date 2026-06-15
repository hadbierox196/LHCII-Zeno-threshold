"""
WEEK 6 — Zeno threshold extraction and statistical analysis.

Fits a sigmoid to the mean IPR(γ) curve and extracts γ_c ± 95% CI
for each (geometry, disorder) combination.  Also runs Mann-Whitney U
test comparing γ_c distributions between geometry classes.

Run: %run scripts/06_threshold_extraction.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import yaml
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.utils import resample

from src.metrics import sigmoid_model, hill_model
from src.visualization import apply_pub_style, save_figure

apply_pub_style()
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

SWEEP_FILE = 'results/sweep_raw.h5'
OUTPUT_CSV = 'results/week6_gamma_c_results.csv'

# ── Config ────────────────────────────────────────────────────────────────────
cfg = {}
try:
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f).get('threshold', {})
except FileNotFoundError:
    pass

P0         = cfg.get('sigmoid_p0', [0.8, 2.0, 3.0, 0.07])
MAXFEV     = int(cfg.get('maxfev', 10000))
N_BOOT     = int(cfg.get('n_bootstrap', 1000))
CI_LEVEL   = float(cfg.get('ci_level', 0.95))
ALPHA      = 1.0 - CI_LEVEL


# ── Sigmoid fitting ───────────────────────────────────────────────────────────

def fit_sigmoid(gamma_array, mean_ipr, p0=None, verbose=True):
    """
    Fit sigmoid to mean IPR vs. log(γ).

    Returns (gamma_c, gamma_c_lo, gamma_c_hi, popt, pcov, r_squared)
    or (nan, nan, nan, None, None, nan) on failure.
    """
    log_gamma = np.log(gamma_array)
    if p0 is None:
        # Auto-estimate: centre near median gamma, amplitude = IPR range
        ipr_amp  = float(np.nanmax(mean_ipr) - np.nanmin(mean_ipr))
        log_gc   = float(log_gamma[np.nanargmax(np.gradient(mean_ipr))])
        p0 = [ipr_amp, 2.0, log_gc, float(np.nanmin(mean_ipr))]

    try:
        popt, pcov = curve_fit(sigmoid_model, log_gamma, mean_ipr,
                               p0=p0, maxfev=MAXFEV)
        A, k, log_gc, B = popt
        gamma_c     = float(np.exp(log_gc))
        sigma_log_gc = float(np.sqrt(np.diag(pcov)[2]))
        z = float(np.abs(np.percentile(
            np.random.normal(0, 1, 100000),
            100 * (1 - (1 - CI_LEVEL) / 2)
        )))
        gamma_c_lo  = float(np.exp(log_gc - z * sigma_log_gc))
        gamma_c_hi  = float(np.exp(log_gc + z * sigma_log_gc))

        # R²
        residuals = mean_ipr - sigmoid_model(log_gamma, *popt)
        ss_res    = np.sum(residuals**2)
        ss_tot    = np.sum((mean_ipr - np.mean(mean_ipr))**2)
        r_sq      = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        if verbose:
            print(f'  Sigmoid fit: A={A:.3f} k={k:.3f} γ_c={gamma_c:.1f} cm⁻¹  R²={r_sq:.3f}')

        return gamma_c, gamma_c_lo, gamma_c_hi, popt, pcov, r_sq

    except RuntimeError as e:
        if verbose:
            print(f'  Sigmoid fit failed: {e}')
            print('  Trying Hill function…')
        return _fit_hill_fallback(gamma_array, mean_ipr, verbose)


def _fit_hill_fallback(gamma_array, mean_ipr, verbose=True):
    """Hill-function fallback when sigmoid doesn't converge."""
    from functools import partial
    def hill_wrap(gamma, n, gamma_c):
        return hill_model(gamma, n, gamma_c,
                          float(np.nanmin(mean_ipr)),
                          float(np.nanmax(mean_ipr)))
    try:
        popt, pcov = curve_fit(hill_wrap, gamma_array, mean_ipr,
                               p0=[2.0, float(np.median(gamma_array))],
                               maxfev=MAXFEV)
        n, gamma_c = popt
        sigma_gc   = float(np.sqrt(pcov[1, 1]))
        z          = 1.96
        if verbose:
            print(f'  Hill fit: n={n:.2f}  γ_c={gamma_c:.1f} cm⁻¹')
        return (float(gamma_c),
                float(gamma_c - z * sigma_gc),
                float(gamma_c + z * sigma_gc),
                popt, pcov, np.nan)
    except Exception as e2:
        if verbose:
            print(f'  Hill fit also failed: {e2}')
        return np.nan, np.nan, np.nan, None, None, np.nan


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_gamma_c(ipr_realizations, gamma_array, n_boot=N_BOOT, p0=None):
    """
    Resample disorder realisations (B=n_boot times) and fit sigmoid each time.

    ipr_realizations : array (n_gamma, n_real)
    Returns (boot_gamma_c array, ci_lo, ci_hi)
    """
    boot_gc = []
    for _ in range(n_boot):
        boot_ipr = resample(ipr_realizations.T, n_samples=ipr_realizations.shape[1]).T
        mean_boot = np.nanmean(boot_ipr, axis=1)
        gc, _, _, _, _, _ = fit_sigmoid(gamma_array, mean_boot, p0=p0, verbose=False)
        if np.isfinite(gc):
            boot_gc.append(gc)

    if len(boot_gc) < 10:
        return np.array(boot_gc), np.nan, np.nan

    ci_lo = float(np.percentile(boot_gc, 100 * ALPHA / 2))
    ci_hi = float(np.percentile(boot_gc, 100 * (1 - ALPHA / 2)))
    return np.array(boot_gc), ci_lo, ci_hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('='*60)
    print('WEEK 6 — Zeno Threshold Extraction')
    print('='*60)

    if not os.path.exists(SWEEP_FILE):
        print(f'ERROR: {SWEEP_FILE} not found.  Run 05_parameter_sweep.py first.')
        return

    with h5py.File(SWEEP_FILE, 'r') as f:
        gamma_array  = f['gamma_array'][:]
        sigma_values = f['sigma_values'][:]
        geometries   = [g.decode() for g in f['geometries'][:]]
        ipr_all      = f['ipr'][:]   # shape (n_gamma, n_sigma, n_geom, n_real)

    print(f'\nLoaded: {SWEEP_FILE}')
    print(f'  γ range: {gamma_array[0]:.1f} – {gamma_array[-1]:.1f} cm⁻¹')
    print(f'  Geometries: {geometries}')
    print(f'  Disorder σ: {sigma_values} cm⁻¹')

    results    = []
    gc_by_geom = {g: [] for g in geometries}    # for Mann-Whitney

    fig_fit, ax_fit = plt.subplots(figsize=(7, 5))
    colors = {'symmetric': '#2C6E9E', 'disordered_monomer': '#C25A3D'}
    ls_map = ['-', '--', ':', '-.', (0, (5,1))]

    for geom_i, geometry in enumerate(geometries):
        color = colors.get(geometry, 'green')
        for si, sigma in enumerate(sigma_values):
            ipr_slice = ipr_all[:, si, geom_i, :]   # (n_gamma, n_real)
            mean_ipr  = np.nanmean(ipr_slice, axis=1)
            std_ipr   = np.nanstd(ipr_slice, axis=1)

            print(f'\n── {geometry}  σ={sigma:.0f} cm⁻¹ ──')

            # ── Sigmoid fit ──────────────────────────────────────────────
            gc, gc_lo, gc_hi, popt, pcov, r_sq = fit_sigmoid(
                gamma_array, mean_ipr, verbose=True)

            # ── Bootstrap ────────────────────────────────────────────────
            print(f'  Running bootstrap (N={N_BOOT})…', end=' ', flush=True)
            boot_gc, boot_lo, boot_hi = bootstrap_gamma_c(
                ipr_slice, gamma_array,
                p0=list(popt) if popt is not None else None)
            print(f'done.  Boot 95% CI: [{boot_lo:.1f}, {boot_hi:.1f}] cm⁻¹')

            # ── Pearson r: γ_c vs σ² (computed across sigma loop) ───────
            # (done after all sigma are collected, see below)

            gc_by_geom[geometry].append(gc)

            rec = {
                'geometry':      geometry,
                'sigma_cm1':     float(sigma),
                'gamma_c':       gc,
                'gamma_c_lo_ci': gc_lo,
                'gamma_c_hi_ci': gc_hi,
                'gamma_c_boot_lo': boot_lo,
                'gamma_c_boot_hi': boot_hi,
                'r_squared':     r_sq,
                'n_boot_valid':  len(boot_gc),
            }
            results.append(rec)

            # ── Plot fitted curve ────────────────────────────────────────
            ls = ls_map[si % len(ls_map)]
            label = f'{geometry.replace("_", " ")}  σ={sigma:.0f}'
            ax_fit.semilogx(gamma_array, mean_ipr, alpha=0.3,
                            color=color, ls=ls, lw=1)
            if popt is not None:
                ipr_fit = sigmoid_model(np.log(gamma_array), *popt)
                ax_fit.semilogx(gamma_array, ipr_fit,
                                color=color, ls=ls, lw=1.8, label=label)
            if np.isfinite(gc):
                ax_fit.axvline(gc, color=color, lw=0.8, alpha=0.5)

    ax_fit.set_xlabel('γ  (cm⁻¹)')
    ax_fit.set_ylabel('Mean IPR')
    ax_fit.set_title('Sigmoid fits to IPR(γ)  —  all (geometry, σ) combinations')
    ax_fit.legend(fontsize=7, ncol=2, framealpha=0.85)
    ax_fit.set_ylim(0, 1.05)
    fig_fit.tight_layout()
    fig_fit.savefig('figures/week6_sigmoid_fits.png', dpi=300, bbox_inches='tight')

    # ── Pearson r: γ_c vs σ² for each geometry ──────────────────────────
    df = pd.DataFrame(results)
    for geometry in geometries:
        sub = df[df['geometry'] == geometry].dropna(subset=['gamma_c'])
        if len(sub) >= 3:
            r, p = pearsonr(sub['sigma_cm1']**2, sub['gamma_c'])
            print(f'\nPearson r (γ_c vs σ²)  [{geometry}]: r={r:.3f}  p={p:.4f}')
            df.loc[df['geometry'] == geometry, 'pearson_r_vs_sigma2'] = r
            df.loc[df['geometry'] == geometry, 'pearson_p_vs_sigma2'] = p

    # ── Mann-Whitney U: geometry comparison ──────────────────────────────
    gc_sym  = [x for x in gc_by_geom.get('symmetric', []) if np.isfinite(x)]
    gc_dis  = [x for x in gc_by_geom.get('disordered_monomer', []) if np.isfinite(x)]
    if gc_sym and gc_dis:
        u_stat, p_mw = mannwhitneyu(gc_sym, gc_dis, alternative='two-sided')
        print(f'\nMann-Whitney U test (symmetric vs. disordered): '
              f'U={u_stat:.1f}  p={p_mw:.4f}')
        df['mw_u_stat'] = u_stat
        df['mw_p_value'] = p_mw

    # ── Save CSV ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nResults saved: {OUTPUT_CSV}')

    # ── Figure 2: γ_c vs σ ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for geometry in geometries:
        sub   = df[df['geometry'] == geometry].sort_values('sigma_cm1')
        color = colors.get(geometry, 'green')
        label = geometry.replace('_', ' ')
        ax2.errorbar(sub['sigma_cm1'], sub['gamma_c'],
                     yerr=[sub['gamma_c'] - sub['gamma_c_lo_ci'],
                           sub['gamma_c_hi_ci'] - sub['gamma_c']],
                     fmt='o-', color=color, lw=2, ms=6,
                     capsize=4, label=label)
    ax2.set_xlabel('Disorder strength σ  (cm⁻¹)')
    ax2.set_ylabel('Zeno threshold  γ_c  (cm⁻¹)')
    ax2.set_title('Figure 2 — Zeno threshold vs. disorder strength')
    ax2.legend(framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig('figures/fig2_gamma_c_vs_sigma.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figures/fig2_gamma_c_vs_sigma.svg')
    print('Figure 2 saved.')

    print('\nSummary of γ_c values:')
    print(df[['geometry', 'sigma_cm1', 'gamma_c', 'gamma_c_lo_ci',
              'gamma_c_hi_ci', 'r_squared']].to_string(index=False))
    print('\n✓ Week 6 complete.  Run scripts/07_spatial_localization.py next.')
    plt.show()


if __name__ == '__main__':
    main()
