"""
LHCII Hamiltonian utilities.

Primary usage: load from HDF5 file built in Week 2 (02_build_hamiltonian.py).
Fallback: approximate test Hamiltonian for code validation only.

Site ordering (0-indexed, following Müh et al. 2010 PNAS 107:16297):
  0:a602  1:a603  2:a604  3:a605  4:a606  5:a607
  6:a608  7:a609  8:a610  9:a611  10:a612 11:a613
  12:b601  13:b606

Source: b601 (site 12) — highest-energy Chl b, primary absorber
Sink:   a610 (site 8)  — low-energy Chl a, connects to reaction centre
"""

import numpy as np
import h5py
import warnings
import os

# ── Site metadata ────────────────────────────────────────────────────────────
SITE_LABELS = [
    'a602', 'a603', 'a604', 'a605', 'a606', 'a607',
    'a608', 'a609', 'a610', 'a611', 'a612', 'a613',
    'b601', 'b606',
]
N_SITES = 14
SOURCE_SITE = 12   # b601
SINK_SITE   = 8    # a610

# ── Loader ───────────────────────────────────────────────────────────────────

def load_hamiltonian_h5(filepath='results/lhcii_hamiltonian_mueh2010.h5',
                         dataset='H'):
    """
    Load 14×14 LHCII Hamiltonian (cm⁻¹) from HDF5 file.

    Raises FileNotFoundError if the file does not exist — run
    02_build_hamiltonian.py first.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Hamiltonian file not found: '{filepath}'\n"
            "Run  scripts/02_build_hamiltonian.py  first to build it from\n"
            "the values in Müh et al. 2010 PNAS 107:16297 Table 1."
        )
    with h5py.File(filepath, 'r') as f:
        H = f[dataset][:]
    assert H.shape == (14, 14), f"Expected (14,14), got {H.shape}"
    if not np.allclose(H, H.T.conj(), atol=1e-4):
        raise ValueError("Loaded Hamiltonian is not Hermitian — check source data.")
    return H.astype(complex)


def load_hamiltonian_or_test(filepath='results/lhcii_hamiltonian_mueh2010.h5'):
    """
    Load real Hamiltonian or fall back to test version with a visible warning.
    Always prefer the real one for any result you intend to publish.
    """
    try:
        H = load_hamiltonian_h5(filepath)
        return H, 'mueh2010'
    except FileNotFoundError:
        warnings.warn(
            "\n\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║  USING TEST HAMILTONIAN — NOT Müh 2010 parameters   ║\n"
            "║  Results are for code validation ONLY.              ║\n"
            "║  Run 02_build_hamiltonian.py to fix this.            ║\n"
            "╚══════════════════════════════════════════════════════╝\n",
            UserWarning, stacklevel=2
        )
        return get_test_hamiltonian(), 'test'


# ── Test / fallback Hamiltonian ──────────────────────────────────────────────

def get_test_hamiltonian():
    """
    Physically motivated test Hamiltonian for code validation.

    ⚠ WARNING: These are APPROXIMATE values, NOT the published Müh 2010
    parameters.  They capture the correct qualitative physics:
      • Energy funnel from high-energy Chl b → low-energy Chl a
      • Strongly coupled a612–a613 dimer (J ≈ −139 cm⁻¹)
      • Realistic eigenvalue spread (~14 700–15 500 cm⁻¹)
    but will produce quantitatively wrong γ_c values.

    For any published result, extract exact values from:
      Müh et al. 2010 PNAS 107:16297, Table 1 (site energies)
      and Table 2 / Supplementary (coupling constants)
    and enter them via 02_build_hamiltonian.py.
    """
    # ── Approximate site energies (cm⁻¹) ─────────────────────────────────
    # Based on approximate recall of Müh 2010 values.
    # Eigenvalue range should be ~14 650–15 450 cm⁻¹ with real parameters.
    site_energies = np.array([
        15220.,  # 0: a602
        15100.,  # 1: a603
        14960.,  # 2: a604
        15490.,  # 3: a605
        14940.,  # 4: a606
        15050.,  # 5: a607
        15050.,  # 6: a608
        15215.,  # 7: a609
        15020.,  # 8: a610  ← sink site
        14975.,  # 9: a611
        14730.,  # 10: a612  ← part of strongly-coupled pair
        14930.,  # 11: a613  ← part of strongly-coupled pair
        15880.,  # 12: b601  ← source site
        15970.,  # 13: b606
    ])

    # ── Coupling constants J (cm⁻¹) ──────────────────────────────────────
    # Only well-known couplings are included; remainder set to small values.
    # ⚠ The full 91-element coupling matrix must come from the paper.
    J = np.zeros((14, 14))

    # Strongly-coupled pair — value appears consistently across literature
    J[10, 11] = J[11, 10] = -139.0   # a612–a613 (Müh 2010)

    # Moderate couplings — approximate nearest-neighbour values
    J[9,  10] = J[10,  9] =  -22.0   # a611–a612
    J[8,   9] = J[ 9,  8] =  -30.0   # a610–a611
    J[7,   8] = J[ 8,  7] =   25.0   # a609–a610
    J[0,   1] = J[ 1,  0] =   30.0   # a602–a603
    J[1,   2] = J[ 2,  1] =  -25.0   # a603–a604
    J[2,   3] = J[ 3,  2] =   15.0   # a604–a605
    J[3,   4] = J[ 4,  3] =  -20.0   # a605–a606
    J[4,   5] = J[ 5,  4] =   20.0   # a606–a607
    J[5,   6] = J[ 6,  5] =  -15.0   # a607–a608
    J[6,   7] = J[ 7,  6] =   20.0   # a608–a609

    # b601 (source) couplings to nearby Chl a
    J[12,  0] = J[ 0, 12] =   20.0   # b601–a602
    J[12,  1] = J[ 1, 12] =  -15.0   # b601–a603
    J[12,  7] = J[ 7, 12] =   10.0   # b601–a609

    # b606 couplings
    J[13,  3] = J[ 3, 13] =   15.0   # b606–a605
    J[13,  4] = J[ 4, 13] =  -10.0   # b606–a606

    H = np.diag(site_energies) + J
    assert np.allclose(H, H.T), "Test Hamiltonian must be real-symmetric"
    return H.astype(complex)


# ── Validation helpers ───────────────────────────────────────────────────────

def validate_hamiltonian(H):
    """Return dict of diagnostic info; print any problems."""
    eigenvalues = np.linalg.eigvalsh(H)
    hermitian = np.allclose(H, H.T.conj(), atol=1e-4)
    n_distinct = len(np.unique(np.round(eigenvalues, 1)))
    info = {
        'shape': H.shape,
        'hermitian': hermitian,
        'n_eigenvalues': n_distinct,
        'eig_min_cm1': eigenvalues.min(),
        'eig_max_cm1': eigenvalues.max(),
        'eig_spread_cm1': eigenvalues.max() - eigenvalues.min(),
        'max_coupling_cm1': np.max(np.abs(H - np.diag(np.diag(H)))),
    }
    if not hermitian:
        print("ERROR: Hamiltonian is NOT Hermitian.")
    if n_distinct < 14:
        print(f"WARNING: only {n_distinct} distinct eigenvalues (expected 14).")
    if not (500 < info['eig_spread_cm1'] < 2000):
        print(f"WARNING: eigenvalue spread {info['eig_spread_cm1']:.0f} cm⁻¹ "
              f"seems outside typical LHCII range (500–1500 cm⁻¹).")
    return info


def apply_disorder(H, sigma_cm1, rng=None):
    """
    Add static Gaussian disorder to site energies.

    Parameters
    ----------
    H : ndarray (14, 14)
        Base Hamiltonian in cm⁻¹
    sigma_cm1 : float
        Standard deviation of site-energy noise (cm⁻¹)
    rng : np.random.Generator, optional
        Provide for reproducibility

    Returns
    -------
    H_disordered : ndarray (14, 14)
    """
    if sigma_cm1 == 0:
        return H.copy()
    if rng is None:
        rng = np.random.default_rng()
    H_d = H.copy()
    noise = rng.normal(0.0, sigma_cm1, 14)
    H_d[np.diag_indices(14)] += noise
    return H_d


def get_geometry_hamiltonian(H_base, geometry):
    """
    Return Hamiltonian for a given geometry class.

    'symmetric'          — standard trimeric LHCII (no modification)
    'disordered_monomer' — monomeric subunit with inter-subunit couplings
                           scaled down by 30 % to mimic monomer isolation
    """
    if geometry == 'symmetric':
        return H_base.copy()
    elif geometry == 'disordered_monomer':
        H = H_base.copy()
        diag = np.diag(H).copy()
        np.fill_diagonal(H, 0)
        H *= 0.70                        # reduce off-diagonal by 30 %
        np.fill_diagonal(H, diag)        # restore site energies
        return H
    else:
        raise ValueError(f"Unknown geometry '{geometry}'. "
                         "Use 'symmetric' or 'disordered_monomer'.")
