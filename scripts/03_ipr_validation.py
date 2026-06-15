"""
WEEK 3 — IPR implementation validation and 5-point preview curve.

Confirms:
  • IPR = 1.0    for fully localised state
  • IPR = 1/14   for fully delocalised state
  • IPR varies monotonically with γ in the source-sink model

Run: %run scripts/03_ipr_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt

from src.hamiltonian import load_hamiltonian_or_test, SOURCE_SITE, SINK_SITE
from src.lindblad import embed_hamiltonian, build_collapse_operators, compute_steady_state
from src.metrics import compute_ipr, validate_ipr_limits

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

KAPPA_IN  = 10.0   # cm⁻¹
KAPPA_OUT = 100.0  # cm⁻¹
GAMMA_5PT = [1.0, 10.0, 100.0, 1000.0, 5000.0]   # cm⁻¹ — 5 representative points


def main():
    print('='*55)
    print('WEEK 3 — IPR Validation')
    print('='*55)

    # ── 1. Validate IPR formula ─────────────────────────────────────────
    validate_ipr_limits()

    # ── 2. Load Hamiltonian ─────────────────────────────────────────────
    H_base, h_source = load_hamiltonian_or_test()
    print(f'\nHamiltonian source: {h_source}')

    # ── 3. Five-point IPR preview ───────────────────────────────────────
    print(f'\nRunning 5-point IPR preview with source (site {SOURCE_SITE}) '
          f'and sink (site {SINK_SITE})…')
    records = []
    for gamma in GAMMA_5PT:
        H_qt  = embed_hamiltonian(H_base)
        c_ops = build_collapse_operators(gamma, KAPPA_IN, KAPPA_OUT,
                                          SOURCE_SITE, SINK_SITE)
        rho_ss = compute_steady_state(H_qt, c_ops)
        if rho_ss is None:
            print(f'  γ={gamma:7.1f} cm⁻¹  →  FAILED (steady state not found)')
            records.append({'gamma_cm1': gamma, 'ipr': np.nan, 'pr': np.nan})
            continue
        ipr, pr, site_pop, total_exc = compute_ipr(rho_ss)
        print(f'  γ={gamma:7.1f} cm⁻¹  →  IPR={ipr:.4f}  PR={pr:.2f}  '
              f'total_exc={total_exc:.4f}')
        records.append({'gamma_cm1': gamma, 'ipr': ipr, 'pr': pr,
                        'total_excited': total_exc})

    df = pd.DataFrame(records)
    csv_out = 'results/week3_ipr_5points.csv'
    df.to_csv(csv_out, index=False)
    print(f'\nSaved: {csv_out}')

    # ── 4. Preview plot ─────────────────────────────────────────────────
    valid = df.dropna()
    if len(valid) < 2:
        print('Not enough valid points to plot.  Check solver.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].semilogx(valid['gamma_cm1'], valid['ipr'], 'o-',
                     color='#2C6E9E', lw=2, ms=7)
    axes[0].axhline(1/14, color='grey', ls='--', lw=1, label='1/14 (delocalised)')
    axes[0].axhline(1.0,  color='grey', ls=':',  lw=1, label='1 (localised)')
    axes[0].set_xlabel('γ  (cm⁻¹)')
    axes[0].set_ylabel('IPR')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('IPR vs. γ  (5-point preview)')
    axes[0].legend(fontsize=8)

    axes[1].semilogx(valid['gamma_cm1'], valid['pr'], 'o-',
                     color='#C25A3D', lw=2, ms=7)
    axes[1].axhline(14, color='grey', ls='--', lw=1, label='14 (delocalised)')
    axes[1].axhline(1,  color='grey', ls=':',  lw=1, label='1 (localised)')
    axes[1].set_xlabel('γ  (cm⁻¹)')
    axes[1].set_ylabel('Participation ratio  1/IPR')
    axes[1].set_title('Participation ratio vs. γ')
    axes[1].legend(fontsize=8)

    fig.suptitle('Week 3 — IPR preview  (source-sink model)', y=1.01)
    fig.tight_layout()
    out = 'figures/week3_ipr_preview.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {out}')

    # ── 5. Diagnosis ─────────────────────────────────────────────────────
    ipr_range = valid['ipr'].max() - valid['ipr'].min()
    if ipr_range < 0.05:
        print('\n⚠  IPR range < 0.05 — source/sink may be too weak or missing.')
        print('   Check that kappa_in and kappa_out are both > 0 in config.yaml.')
    else:
        print(f'\n✓ IPR spans {ipr_range:.3f} — source-sink model is working.')
        print('  Ready for full sweep (Week 5).')
    plt.show()


if __name__ == '__main__':
    main()
