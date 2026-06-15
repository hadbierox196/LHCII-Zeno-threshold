"""
WEEK 7 — Spatial exciton density maps on LHCII molecular geometry.

Extracts Mg atom coordinates from PDB 1RWT (one per chlorophyll),
runs Lindblad at three γ values (below, at, above γ_c), and
produces a three-panel bubble map.

PDB file download:
  !wget -q https://files.rcsb.org/download/1RWT.pdb -O data/1RWT.pdb

Run: %run scripts/07_spatial_localization.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py

from src.hamiltonian import (load_hamiltonian_or_test, SOURCE_SITE, SINK_SITE,
                              SITE_LABELS)
from src.lindblad import embed_hamiltonian, build_collapse_operators, compute_steady_state
from src.metrics import compute_ipr
from src.visualization import apply_pub_style, save_figure

apply_pub_style()
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

PDB_PATH       = 'data/1RWT.pdb'
SWEEP_FILE     = 'results/sweep_raw.h5'
GAMMA_C_FILE   = 'results/week6_gamma_c_results.csv'
KAPPA_IN       = 10.0
KAPPA_OUT      = 100.0
GEOMETRY       = 'symmetric'


# ── PDB coordinate extraction ─────────────────────────────────────────────────

def extract_mg_coords(pdb_path):
    """
    Extract Mg atom positions from LHCII PDB file.

    Returns ndarray (N_MG, 3) of (x, y, z) in Å.

    Expects 14 Mg atoms (one per chlorophyll in the LHCII monomer).
    If you get a different count, check the REMARK/chain records and
    set CHAIN_ID below to restrict to one monomer.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(
            f"PDB file not found: {pdb_path}\n"
            "Download with:\n"
            "  !wget -q https://files.rcsb.org/download/1RWT.pdb -O data/1RWT.pdb"
        )

    try:
        from Bio import PDB
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('LHCII', pdb_path)
        mg_atoms = []
        residue_names = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == 'MG':
                            mg_atoms.append(atom.get_vector().get_array())
                            residue_names.append(residue.get_resname())
            break   # first model only

        coords = np.array(mg_atoms)
        print(f'Found {len(coords)} Mg atoms (residues: {set(residue_names)})')

        if len(coords) != 14:
            print(f'WARNING: expected 14, got {len(coords)}.')
            print('This PDB contains the LHCII trimer (42 Chl) or a different assembly.')
            print('Restricting to first 14 Mg atoms for the monomer.')
            coords = coords[:14]

        return coords

    except ImportError:
        print('BioPython not installed.  Using placeholder coordinates.')
        return _placeholder_coords()


def _placeholder_coords():
    """
    Approximate Mg atom positions (Å) for LHCII monomer.
    These are rough positions for visualisation only.
    Replace with real coordinates from PDB 1RWT.
    """
    print('Using placeholder coordinates — run pip install biopython and download 1RWT.pdb')
    rng = np.random.default_rng(42)
    # LHCII spans roughly 45×45 Å in the membrane plane
    coords = rng.uniform(0, 45, size=(14, 3))
    coords[:, 2] = 0   # project to z=0
    return coords


# ── Run simulations at three γ values ────────────────────────────────────────

def get_gamma_c(geometry=GEOMETRY):
    """Read γ_c (σ=0) from Week 6 output, or use fallback."""
    if os.path.exists(GAMMA_C_FILE):
        df = pd.read_csv(GAMMA_C_FILE)
        row = df[(df['geometry'] == geometry) & (df['sigma_cm1'] == 0)]
        if not row.empty:
            return float(row['gamma_c'].values[0])
    # Fallback: look at sweep to estimate
    if os.path.exists(SWEEP_FILE):
        with h5py.File(SWEEP_FILE, 'r') as f:
            gamma_array = f['gamma_array'][:]
        # Very rough: midpoint of sweep
        return float(gamma_array[len(gamma_array)//2])
    return 300.0   # cm⁻¹ typical default


def run_spatial_sims(H_base, gamma_c):
    """
    Run three Lindblad simulations: γ = γ_c/5, γ_c, γ_c*5.
    Returns dict with site populations for each condition.
    """
    gammas = {
        'below': gamma_c / 5.0,
        'at':    gamma_c,
        'above': gamma_c * 5.0,
    }
    site_pops = {}
    iprs      = {}

    for key, gamma in gammas.items():
        H_qt  = embed_hamiltonian(H_base)
        c_ops = build_collapse_operators(gamma, KAPPA_IN, KAPPA_OUT,
                                          SOURCE_SITE, SINK_SITE)
        rho_ss = compute_steady_state(H_qt, c_ops)
        if rho_ss is None:
            site_pops[key] = np.full(14, np.nan)
            iprs[key] = np.nan
        else:
            ipr, pr, sp, _ = compute_ipr(rho_ss)
            site_pops[key] = sp
            iprs[key] = ipr
        print(f'  γ = {gamma:8.1f} cm⁻¹  →  IPR = {iprs[key]:.4f}')

    return site_pops, iprs, gammas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('='*55)
    print('WEEK 7 — Spatial Localization Topology')
    print('='*55)

    # ── Load Hamiltonian and γ_c ────────────────────────────────────────
    H_base, h_src = load_hamiltonian_or_test()
    gamma_c = get_gamma_c()
    print(f'\nHamiltonian: {h_src}')
    print(f'γ_c (symmetric, σ=0): {gamma_c:.1f} cm⁻¹')

    # ── Simulate ────────────────────────────────────────────────────────
    print('\nRunning simulations at γ < γ_c, ≈ γ_c, > γ_c…')
    site_pops, iprs, gammas = run_spatial_sims(H_base, gamma_c)

    # ── Extract Mg coordinates ──────────────────────────────────────────
    try:
        coords = extract_mg_coords(PDB_PATH)
    except FileNotFoundError as e:
        print(e)
        coords = _placeholder_coords()

    coords_xy = coords[:, :2]   # project onto membrane plane

    # ── Save site population table ──────────────────────────────────────
    df = pd.DataFrame({
        'site': SITE_LABELS,
        'x_A': coords_xy[:, 0],
        'y_A': coords_xy[:, 1],
        'pop_below_gc': site_pops['below'],
        'pop_at_gc':    site_pops['at'],
        'pop_above_gc': site_pops['above'],
    })
    df['ipr_below'] = iprs['below']
    df['ipr_at']    = iprs['at']
    df['ipr_above'] = iprs['above']
    csv_out = 'results/week7_localization_site_table.csv'
    df.to_csv(csv_out, index=False)
    print(f'\nSaved: {csv_out}')

    # ── Three-panel spatial figure ──────────────────────────────────────
    panels = [
        ('below', f'γ = {gammas["below"]:.0f} cm⁻¹\nγ < γ_c  (delocalised)'),
        ('at',    f'γ = {gammas["at"]:.0f} cm⁻¹\nγ ≈ γ_c  (transition)'),
        ('above', f'γ = {gammas["above"]:.0f} cm⁻¹\nγ > γ_c  (Zeno-localised)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    vmax = max(np.nanmax(site_pops[k]) for k in site_pops)

    for ax, (key, subtitle) in zip(axes, panels):
        pop = site_pops[key]
        x, y = coords_xy[:, 0], coords_xy[:, 1]

        # Mark source and sink
        ax.scatter(x[SOURCE_SITE], y[SOURCE_SITE],
                   s=250, marker='*', color='royalblue',
                   zorder=5, label='Source (b601)')
        ax.scatter(x[SINK_SITE], y[SINK_SITE],
                   s=250, marker='D', color='firebrick',
                   zorder=5, label='Sink (a610)')

        sc = ax.scatter(x, y,
                        s=np.clip(pop / vmax, 0, 1) * 600 + 30,
                        c=pop,
                        cmap='YlOrRd',
                        vmin=0, vmax=vmax,
                        edgecolors='k', linewidths=0.6,
                        zorder=3)

        for xi, yi, lbl in zip(x, y, SITE_LABELS):
            ax.annotate(lbl, (xi, yi), fontsize=6.5, ha='center',
                        xytext=(0, 6), textcoords='offset points')

        ax.set_title(f'{subtitle}\nIPR = {iprs[key]:.3f}', fontsize=10)
        ax.set_xlabel('x  (Å)', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('y  (Å)', fontsize=9)
    axes[0].legend(fontsize=7, loc='upper right')

    cbar = plt.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6,
                        label='Normalised site population ρ_nn')
    fig.suptitle('Figure 3 — Spatial exciton density across the Zeno transition\n'
                 f'(source: b601, sink: a610,  γ_c = {gamma_c:.0f} cm⁻¹)', y=1.01)
    fig.tight_layout()

    out = 'figures/fig3_spatial_localization.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig('figures/fig3_spatial_localization.svg')
    print(f'\nFigure 3 saved: {out}')

    # ── Print key finding ───────────────────────────────────────────────
    print('\nTop-3 sites by population at each condition:')
    for key in ('below', 'at', 'above'):
        pop   = site_pops[key]
        idxs  = np.argsort(pop)[::-1][:3]
        sites = [(SITE_LABELS[i], pop[i]) for i in idxs]
        print(f'  {key:5s} (γ={gammas[key]:.0f} cm⁻¹): '
              + '  '.join(f'{s}={v:.3f}' for s, v in sites))

    print('\n✓ Week 7 complete.  Run scripts/08_robustness_controls.py next.')
    plt.show()


if __name__ == '__main__':
    main()
