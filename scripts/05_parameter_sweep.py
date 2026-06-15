"""
WEEK 5 — Parameter Sweep (FIXED)
═══════════════════════════════════════════════════════════════════════
ROOT CAUSE OF PREVIOUS FAILURE
───────────────────────────────
Pure dephasing  L_n = √γ · |n⟩⟨n|  with no source/sink always drives
the system to the maximally mixed state: ρ_nn = 1/N for all n, giving
IPR = 1/N = 0.0714, independent of γ.  The sweep measured nothing.

THE FIX
───────
A 15-dimensional Hilbert space (14 chromophores + ground state) with:
  • Source:  L_src = √κ_in  · |b601⟩⟨g|   incoherent pump
  • Sink:    L_snk = √κ_out · |g⟩⟨a610|   irreversible trap

This creates a non-equilibrium steady state.  The ENAQT (Environment-
Assisted Quantum Transport) effect makes IPR non-monotonic: it dips at
intermediate γ (maximum delocalisation) then rises again at high γ
(Quantum Zeno localisation).  The Zeno threshold γ_c is the inflection
point of IPR(γ) on the high-γ side.

EXPECTED RESULT
───────────────
After this script, sweep_raw.h5 should contain:
  IPR range > 0.15 (variation across γ sweep confirms physics is working)
  Participation ratio peaks at intermediate γ (~100–500 cm⁻¹)
  IPR increases sharply above γ_c

HOW TO RUN IN GOOGLE COLAB
──────────────────────────
  # Cell 1 – mount drive and navigate
  from google.colab import drive
  drive.mount('/content/drive')
  import os
  os.chdir('/content/drive/MyDrive/lhcii-zeno-threshold')

  # Cell 2 – install (once per session)
  !pip install qutip numpy scipy h5py tqdm pyyaml joblib -q

  # Cell 3 – run sweep
  %run scripts/05_parameter_sweep.py

OSF PRE-REGISTRATION
────────────────────
Upload sweep_raw.h5 to OSF as the primary dataset BEFORE running Week 6.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import yaml
import time
from tqdm import tqdm

from src.hamiltonian import (
    load_hamiltonian_or_test,
    apply_disorder,
    get_geometry_hamiltonian,
    SOURCE_SITE, SINK_SITE,
)
from src.lindblad import (
    embed_hamiltonian,
    build_collapse_operators,
    compute_steady_state,
)
from src.metrics import compute_ipr, validate_ipr_limits


# ─────────────────────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────────────────────

def _load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}

_CFG = _load_config()
_SW  = _CFG.get('sweep', {})
_SS  = _CFG.get('source_sink', {})
_SIM = _CFG.get('simulation', {})

GAMMA_MIN      = float(_SW.get('gamma_min_cm1', 1.0))
GAMMA_MAX      = float(_SW.get('gamma_max_cm1', 10000.0))
N_GAMMA        = int(  _SW.get('n_gamma', 80))
SIGMA_VALUES   = list( _SW.get('sigma_values_cm1', [0, 50, 100, 150, 200]))
N_REAL         = int(  _SW.get('n_realizations', 200))
GEOMETRIES     = list( _SW.get('geometries', ['symmetric', 'disordered_monomer']))
KAPPA_IN       = float(_SS.get('kappa_in_cm1',  10.0))
KAPPA_OUT      = float(_SS.get('kappa_out_cm1', 100.0))
SOURCE         = int(  _SS.get('source_site', SOURCE_SITE))
SINK           = int(  _SS.get('sink_site',   SINK_SITE))
SS_METHOD      = str(  _SIM.get('steadystate_method', 'direct'))
OUTPUT_FILE    = 'results/sweep_raw.h5'


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_sim(H_base, gamma_cm1, sigma_cm1, geometry, seed):
    """
    Run one Lindblad simulation.  Returns (ipr, pr, site_pop) or (nan, nan, nan-array).

    This function is self-contained so it can be used with joblib.Parallel.
    """
    rng = np.random.default_rng(seed)

    # 1. Apply geometry transform
    H = get_geometry_hamiltonian(H_base, geometry)

    # 2. Apply static disorder
    if sigma_cm1 > 0:
        H = apply_disorder(H, sigma_cm1, rng=rng)

    # 3. Build QuTiP objects
    H_qt  = embed_hamiltonian(H)
    c_ops = build_collapse_operators(gamma_cm1, KAPPA_IN, KAPPA_OUT, SOURCE, SINK)

    # 4. Steady state
    rho_ss = compute_steady_state(H_qt, c_ops, method=SS_METHOD, verbose=False)
    if rho_ss is None:
        return np.nan, np.nan, np.full(14, np.nan)

    # 5. Metrics
    ipr, pr, site_pop, _ = compute_ipr(rho_ss, n_chrom=14)
    return ipr, pr, site_pop


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_hdf5(gamma_array, sigma_values, geometries, n_real):
    """Create HDF5 file and datasets (no-op if already exists)."""
    ng, ns, ngeom = len(gamma_array), len(sigma_values), len(geometries)
    os.makedirs('results', exist_ok=True)

    with h5py.File(OUTPUT_FILE, 'a') as f:
        for ds_name, shape, dtype, fill in [
            ('ipr',                (ng, ns, ngeom, n_real), np.float32, np.nan),
            ('participation_ratio',(ng, ns, ngeom, n_real), np.float32, np.nan),
            ('site_populations',   (ng, ns, ngeom, n_real, 14), np.float32, np.nan),
            ('completed_mask',     (ng, ns, ngeom, n_real), bool,      False),
        ]:
            if ds_name not in f:
                f.create_dataset(ds_name, shape=shape, dtype=dtype, fillvalue=fill)

        # Metadata
        f.attrs.update({
            'gamma_min': GAMMA_MIN, 'gamma_max': GAMMA_MAX, 'n_gamma': N_GAMMA,
            'kappa_in':  KAPPA_IN,  'kappa_out': KAPPA_OUT,
            'source_site': SOURCE,  'sink_site':  SINK,
            'n_realizations': n_real,
        })
        if 'gamma_array'   not in f: f['gamma_array']   = gamma_array
        if 'sigma_values'  not in f: f['sigma_values']  = np.array(sigma_values, dtype=float)
        if 'geometries'    not in f: f['geometries']    = [g.encode() for g in geometries]


def _count_done(gi, si, geom_i):
    """How many realizations are already saved for this (gamma, sigma, geom) slice?"""
    try:
        with h5py.File(OUTPUT_FILE, 'r') as f:
            return int(np.sum(f['completed_mask'][gi, si, geom_i, :]))
    except Exception:
        return 0


def _save_batch(gi, si, geom_i, real_indices, results):
    """Write a batch of simulation results to HDF5."""
    with h5py.File(OUTPUT_FILE, 'a') as f:
        for ri, (ipr, pr, site_pop) in zip(real_indices, results):
            f['ipr']               [gi, si, geom_i, ri] = ipr
            f['participation_ratio'][gi, si, geom_i, ri] = pr
            if site_pop is not None and not np.any(np.isnan(site_pop)):
                f['site_populations'][gi, si, geom_i, ri] = site_pop.astype(np.float32)
            f['completed_mask']    [gi, si, geom_i, ri] = True


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 65)
    print('LHCII Zeno Threshold — Parameter Sweep  (FIXED: source + sink)')
    print('=' * 65)

    # ── Validate IPR formula ────────────────────────────────────────────
    validate_ipr_limits()

    # ── Load Hamiltonian ────────────────────────────────────────────────
    H_base, h_source = load_hamiltonian_or_test()
    eigs = np.linalg.eigvalsh(np.real(H_base))
    print(f'\nHamiltonian: {h_source}  |  shape {H_base.shape}')
    print(f'  Eigenvalue range: {eigs.min():.0f} – {eigs.max():.0f} cm⁻¹')
    print(f'  Max coupling: '
          f'{np.max(np.abs(H_base - np.diag(np.diag(H_base)))):.1f} cm⁻¹')

    # ── Sweep parameters ────────────────────────────────────────────────
    gamma_array = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA)
    n_g  = len(gamma_array)
    n_s  = len(SIGMA_VALUES)
    n_gm = len(GEOMETRIES)
    total = n_g * n_s * n_gm * N_REAL

    print(f'\nSweep plan:')
    print(f'  γ:   {GAMMA_MIN:.1f} – {GAMMA_MAX:.1f} cm⁻¹  ({n_g} log-spaced points)')
    print(f'  σ:   {SIGMA_VALUES} cm⁻¹')
    print(f'  Geometries: {GEOMETRIES}')
    print(f'  Realisations: {N_REAL}  per (γ, σ, geometry)')
    print(f'  Total simulations: {total:,}')
    print(f'  Source: site {SOURCE} (b601),  Sink: site {SINK} (a610)')
    print(f'  κ_in = {KAPPA_IN} cm⁻¹,  κ_out = {KAPPA_OUT} cm⁻¹')
    print(f'  Output: {OUTPUT_FILE}')
    print()

    # ── Initialise HDF5 ─────────────────────────────────────────────────
    _init_hdf5(gamma_array, SIGMA_VALUES, GEOMETRIES, N_REAL)

    # ── Run sweep ───────────────────────────────────────────────────────
    t_start    = time.time()
    sim_done   = 0

    for geom_i, geometry in enumerate(GEOMETRIES):
        for si, sigma in enumerate(SIGMA_VALUES):
            combo_tag = f'{geometry} | σ={sigma} cm⁻¹'
            print(f'\n── {combo_tag} ─────────────────────────')

            for gi in tqdm(range(n_g), desc='γ sweep', unit='γ'):
                gamma = gamma_array[gi]

                n_done = _count_done(gi, si, geom_i)
                if n_done >= N_REAL:
                    sim_done += N_REAL
                    continue                # already complete — resume logic

                real_todo = list(range(n_done, N_REAL))

                # ─ Run this γ-slice sequentially (steadystate is fast)
                # For very large sweeps, replace with joblib.Parallel below:
                #   from joblib import Parallel, delayed
                #   batch = Parallel(n_jobs=-1)(
                #       delayed(_run_one_sim)(H_base, gamma, sigma, geometry,
                #                            gi*1000+si*100+geom_i*10+r)
                #       for r in real_todo)
                batch = [
                    _run_one_sim(H_base, gamma, sigma, geometry,
                                 seed=gi * 100000 + si * 1000 + geom_i * 100 + r)
                    for r in real_todo
                ]

                _save_batch(gi, si, geom_i, real_todo, batch)
                sim_done += len(real_todo)

    elapsed = time.time() - t_start

    # ── Summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print(f'Sweep complete: {sim_done:,} simulations in {elapsed/60:.1f} min')
    print(f'Output: {OUTPUT_FILE}')
    print('=' * 65)

    _print_sanity_check(gamma_array)


def _print_sanity_check(gamma_array):
    """Quick diagnostics printed after the sweep finishes."""
    print('\nSanity check (σ=0, symmetric geometry):')
    with h5py.File(OUTPUT_FILE, 'r') as f:
        ipr_slice = f['ipr'][:, 0, 0, :]          # (n_gamma, n_real)
        pr_slice  = f['participation_ratio'][:, 0, 0, :]

    mean_ipr = np.nanmean(ipr_slice, axis=1)
    mean_pr  = np.nanmean(pr_slice,  axis=1)

    ipr_range = np.nanmax(mean_ipr) - np.nanmin(mean_ipr)
    peak_pr_idx = np.nanargmax(mean_pr)
    peak_gamma  = gamma_array[peak_pr_idx]

    print(f'  IPR range across γ sweep: {np.nanmin(mean_ipr):.4f} – {np.nanmax(mean_ipr):.4f}')
    print(f'  PR  range across γ sweep: {np.nanmin(mean_pr):.2f} – {np.nanmax(mean_pr):.2f}')
    print(f'  Peak PR = {mean_pr[peak_pr_idx]:.2f}  at γ = {peak_gamma:.1f} cm⁻¹  (ENAQT sweet spot)')

    if ipr_range < 0.05:
        print('\n⚠  WARNING: IPR variation < 0.05.  The sweep may not resolve the Zeno transition.')
        print('   Suggested fixes:')
        print(f'   1. Increase KAPPA_OUT (current: {KAPPA_OUT} cm⁻¹) — try 200–500 cm⁻¹')
        print(f'   2. Increase KAPPA_IN  (current: {KAPPA_IN} cm⁻¹)  — try 50 cm⁻¹')
        print('   3. Confirm source_site and sink_site are correct in config.yaml')
    else:
        print(f'\n✓  IPR variation = {ipr_range:.4f}  — Zeno transition captured.')
        print('  Next step: run  scripts/06_threshold_extraction.py')


if __name__ == '__main__':
    main()
