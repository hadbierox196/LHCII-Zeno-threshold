"""
Publication-quality plotting utilities.

All functions return (fig, ax) or (fig, axes) so callers can further customise.
Figures are saved to the 'figures/' directory by default.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Global style ─────────────────────────────────────────────────────────────

def apply_pub_style(dpi=300, font_size=11, line_width=1.4):
    """Apply publication-quality matplotlib settings."""
    matplotlib.rcParams.update({
        'font.size':        font_size,
        'axes.linewidth':   line_width,
        'axes.titlesize':   font_size + 1,
        'axes.labelsize':   font_size,
        'xtick.labelsize':  font_size - 1,
        'ytick.labelsize':  font_size - 1,
        'legend.fontsize':  font_size - 1,
        'figure.dpi':       dpi,
        'savefig.bbox':     'tight',
        'savefig.dpi':      dpi,
        'lines.linewidth':  line_width,
        'font.family':      'DejaVu Sans',  # safe in Colab; swap for Arial if installed
    })

apply_pub_style()

COLORS = {
    'symmetric':          '#2C6E9E',   # blue
    'disordered_monomer': '#C25A3D',   # red-orange
    'enaqt':              '#3B9E4A',   # green
    'zeno':               '#9B4DCA',   # purple
}

GEOM_LABELS = {
    'symmetric':          'Symmetric (trimeric)',
    'disordered_monomer': 'Disordered monomer',
}


# ── Helper ────────────────────────────────────────────────────────────────────

def save_figure(fig, name, output_dir='figures'):
    """Save figure as both PNG and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
    print(f"  Saved: {output_dir}/{name}.png  +  .svg")


# ── Week 1: dimer validation ─────────────────────────────────────────────────

def plot_dimer_validation(t_ps, populations_dict, gamma_values_ps1):
    """
    Three-panel ρ₁₁(t) for three representative γ values.

    Parameters
    ----------
    t_ps : array (T,)
    populations_dict : dict {gamma: rho11_array}  (3 entries)
    gamma_values_ps1 : list of 3 floats

    Returns  fig, axes (1×3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    labels = ['(a)', '(b)', '(c)']
    descriptions = ['Coherent\n(low γ)', 'Intermediate γ', 'Incoherent\n(high γ)']

    for ax, (gamma, label, desc) in zip(axes,
            zip(gamma_values_ps1, labels, descriptions)):
        rho11 = populations_dict[gamma]
        ax.plot(t_ps, rho11, color='#2C6E9E', lw=1.6)
        ax.set_title(f'{label}  γ = {gamma:.3g} ps⁻¹\n{desc}', fontsize=10)
        ax.set_xlabel('Time (ps)')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

    axes[0].set_ylabel('Population ρ₁₁(t)')
    fig.suptitle('Dimer validation: Lindblad limiting cases', y=1.02)
    fig.tight_layout()
    return fig, axes


# ── Week 3 / Figure 1: IPR vs. γ curve ───────────────────────────────────────

def plot_ipr_curve(gamma_array, mean_ipr_dict, std_ipr_dict=None,
                   gamma_c_dict=None, title='Figure 1 — IPR vs. γ'):
    """
    IPR (and participation ratio on twin axis) vs. log γ for both geometries.

    Parameters
    ----------
    gamma_array : array (n_gamma,)  — cm⁻¹
    mean_ipr_dict : dict {geometry: array (n_gamma,)}
    std_ipr_dict : dict or None
    gamma_c_dict : dict or None   {geometry: gamma_c_cm1}
    """
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()

    for geom, color in COLORS.items():
        if geom not in mean_ipr_dict:
            continue
        ipr  = mean_ipr_dict[geom]
        pr   = 1.0 / np.where(ipr > 1e-12, ipr, np.nan)
        lab  = GEOM_LABELS.get(geom, geom)

        ax1.semilogx(gamma_array, ipr, color=color, lw=2.0, label=lab)
        ax2.semilogx(gamma_array, pr,  color=color, lw=2.0, ls='--', alpha=0.4)

        if std_ipr_dict and geom in std_ipr_dict:
            std = std_ipr_dict[geom]
            ax1.fill_between(gamma_array, ipr - std, ipr + std,
                             color=color, alpha=0.15)

        if gamma_c_dict and geom in gamma_c_dict:
            gc = gamma_c_dict[geom]
            ax1.axvline(gc, color=color, lw=1.2, ls=':', alpha=0.8)
            ax1.text(gc * 1.05, 0.95, f'γ_c={gc:.0f}',
                     color=color, fontsize=8, va='top')

    ax1.set_xlabel('Dephasing rate γ  (cm⁻¹)')
    ax1.set_ylabel('IPR', color='black')
    ax2.set_ylabel('Participation ratio  1/IPR', color='grey')
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 15)
    ax1.legend(loc='upper left', framealpha=0.9)

    # Annotate ENAQT region and Zeno regime
    ax1.axvspan(50, 500, alpha=0.05, color='green', label='ENAQT region')
    ax1.axvspan(2000, gamma_array[-1], alpha=0.05, color='purple')
    ax1.text(100, 0.08, 'ENAQT sweet spot', color='green', fontsize=8, alpha=0.8)
    ax1.text(2500, 0.08, 'Zeno regime', color='purple', fontsize=8, alpha=0.8)

    ax1.set_title(title)
    fig.tight_layout()
    return fig, (ax1, ax2)


# ── Week 6 / Figure 2: γ_c vs. disorder strength ─────────────────────────────

def plot_gamma_c_vs_sigma(sigma_values, gamma_c_results, title='Figure 2 — Zeno threshold vs. disorder'):
    """
    γ_c ± 95 % CI as a function of disorder σ for both geometries.

    Parameters
    ----------
    sigma_values : array (n_sigma,)
    gamma_c_results : dict {geometry: {'mean': array, 'ci_lo': array, 'ci_hi': array}}
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for geom, color in COLORS.items():
        if geom not in gamma_c_results:
            continue
        res  = gamma_c_results[geom]
        mean = np.asarray(res['mean'])
        lo   = np.asarray(res['ci_lo'])
        hi   = np.asarray(res['ci_hi'])
        lab  = GEOM_LABELS.get(geom, geom)

        ax.plot(sigma_values, mean, 'o-', color=color, lw=2, label=lab, ms=5)
        ax.fill_between(sigma_values, lo, hi, color=color, alpha=0.2)

    ax.set_xlabel('Disorder strength σ  (cm⁻¹)')
    ax.set_ylabel('Zeno threshold  γ_c  (cm⁻¹)')
    ax.legend(framealpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ── Week 7 / Figure 3: spatial localisation maps ─────────────────────────────

def plot_spatial_localization(coords_xy, site_pops_dict, gamma_labels,
                               site_labels=None,
                               title='Figure 3 — Spatial exciton density'):
    """
    Three-panel bubble map: exciton density at γ < γ_c, ≈ γ_c, > γ_c.

    Parameters
    ----------
    coords_xy : array (14, 2)  — Mg atom x,y coordinates in Å
    site_pops_dict : dict {'below': array(14), 'at': array(14), 'above': array(14)}
    gamma_labels : dict {'below': str, 'at': str, 'above': str}
    site_labels : list of 14 str, optional
    """
    if site_labels is None:
        from src.hamiltonian import SITE_LABELS
        site_labels = SITE_LABELS

    panels = [('below', 'γ < γ_c\n(delocalised)'),
              ('at',    'γ ≈ γ_c\n(transition)'),
              ('above', 'γ > γ_c\n(Zeno-localised)')]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    vmax = max(np.max(v) for v in site_pops_dict.values())

    for ax, (key, subtitle) in zip(axes, panels):
        pop = site_pops_dict[key]
        x, y = coords_xy[:, 0], coords_xy[:, 1]
        sc = ax.scatter(x, y,
                        s=pop / vmax * 800 + 20,
                        c=pop,
                        cmap='hot_r',
                        vmin=0, vmax=vmax,
                        edgecolors='k', linewidths=0.5,
                        zorder=3)
        for xi, yi, lbl in zip(x, y, site_labels):
            ax.annotate(lbl, (xi, yi), fontsize=6, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')
        gval = gamma_labels.get(key, '')
        ax.set_title(f'{subtitle}\nγ = {gval}', fontsize=10)
        ax.set_xlabel('x  (Å)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('y  (Å)')
    plt.colorbar(sc, ax=axes, label='Normalised site population ρ_nn', shrink=0.7)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


# ── Week 8 / Figure 4: robustness panel ──────────────────────────────────────

def plot_bootstrap_distribution(gamma_c_boots, gamma_c_point, ci95,
                                 geometry='symmetric',
                                 title='Figure 4 — Bootstrap γ_c distribution'):
    """
    Histogram of 1000 bootstrap γ_c estimates with 95 % CI shaded.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    color = COLORS.get(geometry, '#2C6E9E')

    ax.hist(gamma_c_boots, bins=40, color=color, alpha=0.75, edgecolor='w', lw=0.4)
    ax.axvline(gamma_c_point, color='black', lw=1.8, label=f'Point estimate: {gamma_c_point:.1f} cm⁻¹')
    ax.axvspan(ci95[0], ci95[1], color=color, alpha=0.25,
               label=f'95 % CI: [{ci95[0]:.1f}, {ci95[1]:.1f}] cm⁻¹')

    ax.set_xlabel('Bootstrap γ_c  (cm⁻¹)')
    ax.set_ylabel('Count')
    ax.legend(framealpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
