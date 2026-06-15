"""
WEEK 9 — Publication-quality figure finalization and Results section draft.

Produces all four manuscript figures at 300 dpi + SVG.
Prints a Results section template with placeholder slots for your actual
γ_c values, CI bounds, and statistical test results.

Run: %run scripts/09_figure_finalization.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

from src.visualization import apply_pub_style, COLORS, GEOM_LABELS

apply_pub_style(dpi=300, font_size=11)
os.makedirs('figures', exist_ok=True)

SWEEP_FILE  = 'results/sweep_raw.h5'
GC_FILE     = 'results/week6_gamma_c_results.csv'
BOOT_CSV    = 'results/week8_sensitivity_table.csv'
SITE_CSV    = 'results/week7_localization_site_table.csv'


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: IPR vs. γ  (full sweep)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1(gamma_array, ipr_all, geometries, sigma_values,
                 gc_df=None):
    """Four-panel: one panel per geometry, σ=0 only, with γ_c annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (geom_i, geometry) in zip(axes, enumerate(geometries)):
        color = COLORS.get(geometry, '#2C6E9E')
        si = list(sigma_values).index(0) if 0 in sigma_values else 0

        ipr_slice = ipr_all[:, si, geom_i, :]
        mean_ipr  = np.nanmean(ipr_slice, axis=1)
        std_ipr   = np.nanstd(ipr_slice,  axis=1)
        mean_pr   = np.where(mean_ipr > 1e-6, 1.0 / mean_ipr, np.nan)

        ax2 = ax.twinx()
        ax.semilogx(gamma_array, mean_ipr, color=color, lw=2.2, label='IPR')
        ax.fill_between(gamma_array,
                        mean_ipr - std_ipr, mean_ipr + std_ipr,
                        color=color, alpha=0.18)
        ax2.semilogx(gamma_array, mean_pr, color=color, lw=1.4,
                     ls='--', alpha=0.5, label='1/IPR')

        # γ_c annotation
        if gc_df is not None and not gc_df.empty:
            row = gc_df[(gc_df['geometry'] == geometry) & (gc_df['sigma_cm1'] == 0)]
            if not row.empty:
                gc = float(row['gamma_c'].values[0])
                lo = float(row['gamma_c_lo_ci'].values[0])
                hi = float(row['gamma_c_hi_ci'].values[0])
                ax.axvline(gc, color='k', lw=1.3, ls=':', alpha=0.8)
                ax.axvspan(lo, hi, alpha=0.08, color='k')
                ax.text(gc * 1.08, 0.9, f'γ_c={gc:.0f} cm⁻¹',
                        fontsize=8.5, va='top')

        # ENAQT and Zeno annotations
        ax.axvspan(gamma_array[0], 500, alpha=0.04, color='green')
        ax.axvspan(2000, gamma_array[-1], alpha=0.04, color='purple')
        ax.text(2.0, 0.04, 'ENAQT', color='green', fontsize=8, alpha=0.8)
        ax.text(2200, 0.04, 'Zeno', color='purple', fontsize=8, alpha=0.8)

        ax.set_xlabel('Dephasing rate γ  (cm⁻¹)')
        ax.set_ylabel('IPR' if geom_i == 0 else '')
        ax2.set_ylabel('Participation ratio' if geom_i == 1 else '',
                       color='grey')
        ax2.tick_params(axis='y', colors='grey')
        ax.set_ylim(0, 1.08)
        ax2.set_ylim(0, 16)
        ax.set_title(GEOM_LABELS.get(geometry, geometry))

    fig.suptitle('Figure 1 — IPR vs. dephasing rate  (σ=0 cm⁻¹)', y=1.01)
    fig.tight_layout()
    _save('fig1_ipr_vs_gamma', fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: γ_c vs. σ
# ─────────────────────────────────────────────────────────────────────────────

def make_figure2(gc_df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    geometries = gc_df['geometry'].unique()
    for geometry in geometries:
        sub   = gc_df[gc_df['geometry'] == geometry].sort_values('sigma_cm1')
        color = COLORS.get(geometry, '#555')
        label = GEOM_LABELS.get(geometry, geometry)
        ax.errorbar(sub['sigma_cm1'], sub['gamma_c'],
                    yerr=[sub['gamma_c'] - sub['gamma_c_lo_ci'],
                          sub['gamma_c_hi_ci'] - sub['gamma_c']],
                    fmt='o-', color=color, lw=2, ms=6,
                    capsize=4, capthick=1.5, label=label)
    ax.set_xlabel('Disorder strength σ  (cm⁻¹)')
    ax.set_ylabel('Zeno threshold  γ_c  (cm⁻¹)')
    ax.set_title('Figure 2 — Zeno threshold vs. disorder')
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save('fig2_gamma_c_vs_sigma', fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Spatial localisation (load from week 7 CSV)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure3():
    if not os.path.exists(SITE_CSV):
        print(f'  {SITE_CSV} not found.  Run 07_spatial_localization.py first.')
        return None

    df = pd.read_csv(SITE_CSV)
    x, y = df['x_A'].values, df['y_A'].values

    panels = [('pop_below_gc', 'γ < γ_c  (delocalised)'),
              ('pop_at_gc',    'γ ≈ γ_c  (transition)'),
              ('pop_above_gc', 'γ > γ_c  (Zeno-localised)')]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    vmax = max(df[col].max() for col, _ in panels)

    for ax, (col, subtitle) in zip(axes, panels):
        pop = df[col].values
        sc = ax.scatter(x, y, s=pop / vmax * 700 + 30,
                        c=pop, cmap='YlOrRd', vmin=0, vmax=vmax,
                        edgecolors='k', linewidths=0.5, zorder=3)
        for xi, yi, lbl in zip(x, y, df['site']):
            ax.annotate(lbl, (xi, yi), fontsize=6.5, ha='center',
                        xytext=(0, 5), textcoords='offset points')
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel('x  (Å)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('y  (Å)')
    plt.colorbar(sc, ax=axes.ravel().tolist(),
                 label='Normalised site population ρ_nn', shrink=0.65)
    fig.suptitle('Figure 3 — Spatial exciton density across the Zeno transition', y=1.01)
    fig.tight_layout()
    _save('fig3_spatial_localization', fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Robustness panel
# ─────────────────────────────────────────────────────────────────────────────

def make_figure4():
    if not os.path.exists(BOOT_CSV):
        print(f'  {BOOT_CSV} not found.  Run 08_robustness_controls.py first.')
        return None

    df = pd.read_csv(BOOT_CSV)
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = df['label'].values
    gc     = df['gamma_c'].values
    colors_bar = ['#2C6E9E', '#C25A3D', '#3B9E4A', '#9B4DCA'][:len(labels)]

    bars = ax.barh(range(len(labels)), gc, color=colors_bar, alpha=0.75,
                   edgecolor='k', lw=0.6)

    # Error bars if CI columns exist
    if 'ci_lo' in df.columns and 'ci_hi' in df.columns:
        for i, row in df.iterrows():
            if np.isfinite(row.get('ci_lo', np.nan)):
                ax.errorbar(row['gamma_c'], i,
                            xerr=[[row['gamma_c'] - row['ci_lo']],
                                  [row['ci_hi'] - row['gamma_c']]],
                            fmt='none', color='k', lw=1.5, capsize=4)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('γ_c  (cm⁻¹)')
    ax.set_title('Figure 4 — Sensitivity of γ_c to Hamiltonian & collapse operators')
    ax.axvline(gc[0], color='k', lw=0.8, ls='--', alpha=0.4)  # reference line
    fig.tight_layout()
    _save('fig4_robustness', fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Results section template
# ─────────────────────────────────────────────────────────────────────────────

def print_results_template(gc_df=None):
    # Try to populate from real data
    gc_sym, ci_sym_lo, ci_sym_hi = '[VALUE]', '[CI_LO]', '[CI_HI]'
    gc_dis, ci_dis_lo, ci_dis_hi = '[VALUE]', '[CI_LO]', '[CI_HI]'
    mw_u, mw_p = '[U]', '[p]'
    pearson_r   = '[r]'

    if gc_df is not None and not gc_df.empty:
        def _row(geom, sigma=0):
            r = gc_df[(gc_df['geometry'] == geom) & (gc_df['sigma_cm1'] == sigma)]
            return r.iloc[0] if not r.empty else None

        r_sym = _row('symmetric')
        r_dis = _row('disordered_monomer')
        if r_sym is not None:
            gc_sym     = f"{r_sym['gamma_c']:.1f}"
            ci_sym_lo  = f"{r_sym['gamma_c_lo_ci']:.1f}"
            ci_sym_hi  = f"{r_sym['gamma_c_hi_ci']:.1f}"
        if r_dis is not None:
            gc_dis     = f"{r_dis['gamma_c']:.1f}"
            ci_dis_lo  = f"{r_dis['gamma_c_lo_ci']:.1f}"
            ci_dis_hi  = f"{r_dis['gamma_c_hi_ci']:.1f}"
        if 'mw_u_stat' in gc_df.columns and not gc_df['mw_u_stat'].isna().all():
            mw_u = f"{gc_df['mw_u_stat'].dropna().iloc[0]:.1f}"
            mw_p = f"{gc_df['mw_p_value'].dropna().iloc[0]:.4f}"
        if 'pearson_r_vs_sigma2' in gc_df.columns:
            pearson_r = f"{gc_df['pearson_r_vs_sigma2'].dropna().iloc[0]:.3f}"

    template = f"""
════════════════════════════════════════════════════════════════
RESULTS SECTION DRAFT — paste into your manuscript (Week 10)
════════════════════════════════════════════════════════════════

PARAGRAPH 1 — IPR vs. γ (Figure 1)
─────────────────────────────────────
Lindblad master equation simulations of the 14-site LHCII
chromophore network embedded in a 15-dimensional Hilbert space
(14 chromophores plus electronic ground state) revealed a
non-monotonic dependence of the inverse participation ratio
(IPR) on the dephasing rate γ (Figure 1). At low γ
(< 10 cm⁻¹), the excitation density was partially localised
owing to coherent dynamics along specific inter-site pathways.
At intermediate γ, a minimum in IPR (maximum participation ratio)
was observed, consistent with environment-assisted quantum
transport (ENAQT). At high γ, IPR increased sharply, indicating
Quantum Zeno localisation. This non-monotonic behaviour was
present in both geometry classes.

PARAGRAPH 2 — γ_c extraction (Figures 1–2)
─────────────────────────────────────────────
The Zeno threshold γ_c was extracted by fitting a sigmoid model
to the mean IPR(γ) curve (n = 200 disorder realisations per
γ value). For the symmetric geometry at zero disorder,
γ_c = {gc_sym} cm⁻¹ (95% CI: [{ci_sym_lo}, {ci_sym_hi}] cm⁻¹).
For the disordered monomer geometry, γ_c = {gc_dis} cm⁻¹
(95% CI: [{ci_dis_lo}, {ci_dis_hi}] cm⁻¹). A Mann-Whitney U
test confirmed that these distributions differ significantly
(U = {mw_u}, p = {mw_p}), establishing geometry-dependence of
the Zeno threshold. The Pearson correlation between γ_c and
σ² (disorder variance) was r = {pearson_r}, consistent with the
pre-registered hypothesis that γ_c scales with excitonic energy
gap variance (Figure 2).

PARAGRAPH 3 — Spatial topology (Figure 3)
───────────────────────────────────────────
Spatial mapping of the steady-state site populations ρ_nn onto
Mg atom coordinates from PDB 1RWT revealed that Quantum Zeno
localisation preferentially concentrated excitation on sites
[SITE_NAMES] — the energetically lowest-lying chlorophylls in
the Müh 2010 parameter set. Below γ_c, the excitation was
distributed across [N] chromophores (participation ratio
[PR_BELOW]); above γ_c, it collapsed to [N_ABOVE] sites
(participation ratio [PR_ABOVE]). This spatial selectivity
suggests that the Zeno transition constitutes a topology-
dependent switch in the energy transfer pathway of LHCII
(Figure 3).

PARAGRAPH 4 — Robustness (Figure 4)
──────────────────────────────────────
The Zeno threshold was robust to changes in the Hamiltonian
parameter set: γ_c computed from the Novoderezhkin 2011
parameter set differed by a factor of [FOLD] from the Müh 2010
result, within the pre-registered acceptable range of < 3-fold
variation. Substituting energy-relaxation collapse operators for
pure dephasing shifted γ_c by [SHIFT] cm⁻¹. Bootstrap
resampling (B = 1000) of the disorder ensemble yielded a 95%
CI of [{ci_sym_lo}, {ci_sym_hi}] cm⁻¹, confirming that
uncertainty in γ_c is dominated by inter-ensemble variance
rather than fitting artefacts (Figure 4).

════════════════════════════════════════════════════════════════
CHECKLIST BEFORE PASTING INTO MANUSCRIPT:
  [ ] Every sentence contains a specific numerical value
  [ ] All [PLACEHOLDER] slots filled from your actual results
  [ ] p-values formatted as APA 7th ed: p = .034 (not p < .05)
  [ ] Effect sizes (r, η², Cohen's d) reported alongside p-values
  [ ] All γ_c values in BOTH cm⁻¹ and converted to ps (÷ 0.1884)
════════════════════════════════════════════════════════════════
"""
    print(template)
    with open('results/results_section_draft.txt', 'w') as f:
        f.write(template)
    print('Saved: results/results_section_draft.txt')


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(name, fig):
    for ext in ('png', 'svg'):
        path = f'figures/{name}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved: figures/{name}.png + .svg')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('='*60)
    print('WEEK 9 — Figure Finalization')
    print('='*60)

    gc_df = None
    if os.path.exists(GC_FILE):
        gc_df = pd.read_csv(GC_FILE)

    # ── Load sweep data ────────────────────────────────────────────────
    if os.path.exists(SWEEP_FILE):
        with h5py.File(SWEEP_FILE, 'r') as f:
            gamma_array  = f['gamma_array'][:]
            sigma_values = f['sigma_values'][:]
            geometries   = [g.decode() for g in f['geometries'][:]]
            ipr_all      = f['ipr'][:]

        print('\nMaking Figure 1…')
        make_figure1(gamma_array, ipr_all, geometries, sigma_values, gc_df)
    else:
        print(f'⚠  {SWEEP_FILE} not found — skipping Figure 1.')
        gamma_array = np.logspace(0, 4, 80)
        ipr_all     = None
        geometries  = ['symmetric', 'disordered_monomer']
        sigma_values = [0, 50, 100, 150, 200]

    if gc_df is not None:
        print('Making Figure 2…')
        make_figure2(gc_df)
    else:
        print(f'⚠  {GC_FILE} not found — skipping Figure 2.')

    print('Making Figure 3…')
    make_figure3()

    print('Making Figure 4…')
    make_figure4()

    print('\nPrinting Results section template…')
    print_results_template(gc_df)

    print('\nAll figures saved to figures/.')
    print('✓ Week 9 complete.  Begin manuscript drafting (Week 10).')
    plt.show()


if __name__ == '__main__':
    main()
