"""
RUN THIS FIRST — Quick diagnostic to confirm the Week 5 fix works.

Takes ~30 seconds. Runs 7 gamma values and prints IPR.
If IPR varies, the source-sink model is working.
If IPR is flat at 0.0714, something is wrong (see troubleshooting below).

Run in Colab:
  %run scripts/00_quick_diagnostic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qutip as qt
from src.hamiltonian import load_hamiltonian_or_test, SOURCE_SITE, SINK_SITE
from src.lindblad import (embed_hamiltonian, build_collapse_operators,
                           compute_steady_state, CM1_TO_RADPS)
from src.metrics import compute_ipr, validate_ipr_limits

KAPPA_IN  = 10.0    # cm^-1
KAPPA_OUT = 100.0   # cm^-1
GAMMAS    = [1, 10, 50, 100, 300, 1000, 5000]   # cm^-1


def main():
    print('=' * 60)
    print('QUICK DIAGNOSTIC — Source-Sink Model Verification')
    print('=' * 60)

    # 1. Validate IPR formula
    validate_ipr_limits()

    # 2. Load Hamiltonian
    H_base, h_src = load_hamiltonian_or_test()
    print(f'\nHamiltonian: {h_src}')
    eigs = np.linalg.eigvalsh(np.real(H_base))
    print(f'Eigenvalue range: {eigs.min():.0f} – {eigs.max():.0f} cm⁻¹')

    # 3. Show what BROKEN model gives (pure dephasing, no source/sink)
    print('\n── BROKEN model (pure dephasing, no source/sink) ──')
    print('All IPR values should be 0.0714 (maximally mixed state):')
    H_qt = embed_hamiltonian(H_base)
    for gamma in [10, 100, 1000]:
        # Only dephasing, no source or sink
        c_ops_broken = build_collapse_operators(
            gamma_cm1=gamma,
            kappa_in_cm1=0.0,    # ← no source
            kappa_out_cm1=0.0,   # ← no sink
        )
        rho_ss = compute_steady_state(H_qt, c_ops_broken, verbose=False)
        if rho_ss is not None:
            ipr, pr, _, _ = compute_ipr(rho_ss)
            flag = '← confirms bug' if abs(ipr - 1/14) < 0.005 else '← unexpected'
            print(f'  γ = {gamma:5.0f} cm⁻¹  IPR = {ipr:.4f}  PR = {pr:.2f}  {flag}')

    # 4. Show what FIXED model gives (source + sink)
    print(f'\n── FIXED model (source: b601={SOURCE_SITE}, sink: a610={SINK_SITE}) ──')
    print(f'κ_in = {KAPPA_IN} cm⁻¹,  κ_out = {KAPPA_OUT} cm⁻¹')
    print(f'{"γ (cm⁻¹)":>12}  {"IPR":>8}  {"PR (1/IPR)":>12}  {"Excited frac":>14}  Status')
    print('─' * 60)

    iprs, prs = [], []
    all_ok = True
    for gamma in GAMMAS:
        H_qt  = embed_hamiltonian(H_base)
        c_ops = build_collapse_operators(gamma, KAPPA_IN, KAPPA_OUT,
                                          SOURCE_SITE, SINK_SITE)
        rho_ss = compute_steady_state(H_qt, c_ops, verbose=False)
        if rho_ss is None:
            print(f'  {gamma:>10.0f}  FAILED — steady state not found')
            all_ok = False
            continue
        ipr, pr, site_pop, total_exc = compute_ipr(rho_ss)
        iprs.append(ipr)
        prs.append(pr)
        status = '✓' if np.isfinite(ipr) and ipr > 1e-6 else '⚠ CHECK'
        print(f'  {gamma:>10.0f}  {ipr:>8.4f}  {pr:>12.2f}  {total_exc:>14.4f}  {status}')

    # 5. Verdict
    ipr_range = max(iprs) - min(iprs) if iprs else 0
    print('─' * 60)
    print(f'\nIPR range: {min(iprs):.4f} – {max(iprs):.4f}  (variation = {ipr_range:.4f})')

    if ipr_range < 0.05:
        print('\n⚠  PROBLEM: IPR range < 0.05.  The fix is not working correctly.')
        print('   Troubleshooting checklist:')
        print('   1. Confirm kappa_in > 0 and kappa_out > 0 in config.yaml')
        print(f'      Current: kappa_in={KAPPA_IN}, kappa_out={KAPPA_OUT}')
        print('   2. Confirm SOURCE_SITE and SINK_SITE are different indices:')
        print(f'      source={SOURCE_SITE}, sink={SINK_SITE}')
        print('   3. Confirm the Hamiltonian is loaded correctly (eigenvalues above)')
        print('   4. Try increasing kappa_out to 200–500 cm⁻¹ in config.yaml')
    elif max(prs) < 5.0:
        print('\n⚠  WARNING: Max participation ratio < 5. ENAQT effect may be weak.')
        print('   Consider increasing kappa_out or adjusting source/sink sites.')
    else:
        pr_peak_idx = int(np.argmax(prs))
        gamma_peak  = GAMMAS[pr_peak_idx]
        print(f'\n✓ SOURCE-SINK FIX IS WORKING.')
        print(f'  Peak participation ratio: {max(prs):.2f} at γ ≈ {gamma_peak} cm⁻¹')
        print(f'  This is the ENAQT sweet spot.')
        print(f'\n  The IPR curve is non-monotonic:')
        print(f'    Low γ  → partial localisation (coherent regime)')
        print(f'    γ ≈ {gamma_peak} cm⁻¹ → maximum delocalisation (ENAQT)')
        print(f'    High γ → Quantum Zeno localisation')
        print(f'\n  Ready to run the full sweep: %run scripts/05_parameter_sweep.py')

    # 6. Quick plot
    if iprs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].semilogx(GAMMAS[:len(iprs)], iprs, 'o-',
                         color='#2C6E9E', lw=2, ms=8)
        axes[0].axhline(1/14, color='grey', ls='--', lw=1,
                        label='1/14 (maximally mixed)')
        axes[0].set_xlabel('γ  (cm⁻¹)')
        axes[0].set_ylabel('IPR')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title('FIXED model: IPR vs. γ')
        axes[0].legend()

        axes[1].semilogx(GAMMAS[:len(prs)], prs, 'o-',
                         color='#C25A3D', lw=2, ms=8)
        axes[1].axhline(14, color='grey', ls='--', lw=1,
                        label='14 (maximally delocalised)')
        axes[1].set_xlabel('γ  (cm⁻¹)')
        axes[1].set_ylabel('Participation ratio  1/IPR')
        axes[1].set_title('ENAQT peak visible?')
        axes[1].legend()

        fig.suptitle('Diagnostic: source-sink model verification', y=1.01)
        fig.tight_layout()
        os.makedirs('figures', exist_ok=True)
        fig.savefig('figures/00_diagnostic.png', dpi=150, bbox_inches='tight')
        print('\nDiagnostic plot saved: figures/00_diagnostic.png')

        try:
            plt.show()
        except Exception:
            pass   # non-interactive backend in some Colab configs


if __name__ == '__main__':
    main()
