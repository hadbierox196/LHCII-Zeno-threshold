"""
Observables for Zeno threshold quantification.

IPR (Inverse Participation Ratio)
    IPR = Σ_n ρ_nn² / (Σ_n ρ_nn)²

    Range: [1/N, 1]
    IPR = 1/N  → fully delocalised (all sites equally populated)
    IPR = 1    → fully localised (one site holds all population)

Participation Ratio (PR = 1/IPR)
    Range: [1, N]
    PR = N  → delocalised
    PR = 1  → localised

The Zeno transition appears as:
    • Non-monotonic PR(γ): peak at ENAQT sweet spot, then decline
    • IPR(γ): valley at ENAQT, then increase toward 1 at high γ
    • γ_c is the inflection point of IPR(γ) on the high-γ side
"""

import numpy as np

N_CHROM = 14


# ── Core IPR ─────────────────────────────────────────────────────────────────

def compute_ipr(rho_ss, n_chrom=N_CHROM):
    """
    Compute IPR and related quantities from a QuTiP steady-state Qobj.

    Parameters
    ----------
    rho_ss : Qobj
        Steady-state density matrix (15×15 for 15-site model).
    n_chrom : int
        Number of chromophore sites (first n_chrom rows/cols).

    Returns
    -------
    ipr : float
        Inverse participation ratio ∈ [1/n_chrom, 1].
    pr : float
        Participation ratio = 1/IPR ∈ [1, n_chrom].
    site_pop : ndarray (n_chrom,)
        Normalised population on each chromophore (sums to 1).
    total_excited : float
        Total population in the excited manifold (< 1 due to ground state).
    """
    diag = np.real(np.diag(rho_ss.full()))

    chrom_pop   = diag[:n_chrom]            # chromophore populations only
    total_exc   = float(np.sum(chrom_pop))  # fraction of excitation in system

    if total_exc < 1e-12:
        nan_arr = np.full(n_chrom, np.nan)
        return np.nan, np.nan, nan_arr, 0.0

    site_pop = chrom_pop / total_exc        # normalise to sum = 1
    ipr      = float(np.sum(site_pop ** 2))
    pr       = 1.0 / ipr if ipr > 1e-12 else np.inf

    return ipr, pr, site_pop, total_exc


def ipr_from_array(rho_diag, n_chrom=N_CHROM):
    """
    Compute IPR from a raw diagonal array (faster, no Qobj overhead).

    Parameters
    ----------
    rho_diag : array-like, shape (≥ n_chrom,)
        Diagonal of density matrix.

    Returns
    -------
    ipr : float
    """
    pop   = np.abs(np.asarray(rho_diag)[:n_chrom])
    total = np.sum(pop)
    if total < 1e-12:
        return np.nan
    return float(np.sum((pop / total) ** 2))


# ── Validation checks ─────────────────────────────────────────────────────────

def validate_ipr_limits(n_chrom=N_CHROM):
    """
    Confirm IPR formula gives correct values at the two extremes.
    Call this once on start-up to catch any formula regressions.
    """
    import qutip as qt

    # Fully localised: all population on site 0
    rho_loc = qt.Qobj(np.diag(
        [1.0] + [0.0] * 13 + [0.0]   # 15-dim: sites 0–13 + ground
    ))
    ipr, pr, _, _ = compute_ipr(rho_loc)
    assert abs(ipr - 1.0) < 1e-9,  f"Localised IPR should be 1.0, got {ipr}"
    assert abs(pr  - 1.0) < 1e-9,  f"Localised PR  should be 1.0, got {pr}"

    # Fully delocalised: uniform population across all 14 chromophores
    rho_deloc = qt.Qobj(np.diag(
        [1.0 / 14] * 14 + [0.0]      # 15-dim
    ))
    ipr, pr, _, _ = compute_ipr(rho_deloc)
    assert abs(ipr - 1.0 / 14) < 1e-9, f"Delocalised IPR should be 1/14={1/14:.4f}, got {ipr}"
    assert abs(pr  - 14.0)     < 1e-9, f"Delocalised PR  should be 14, got {pr}"

    print("✓ IPR validation passed: localised=1.0, delocalised=1/14=0.0714")
    return True


# ── Sigmoid fit model (for threshold extraction in Week 6) ───────────────────

def sigmoid_model(log_gamma, A, k, log_gamma_c, B):
    """
    Sigmoid function for fitting IPR(γ) curve.

    IPR(γ) = A / (1 + exp(−k · (log γ − log γ_c))) + B

    Parameters
    ----------
    log_gamma : array-like
        Natural log of gamma values.
    A : float
        Amplitude (IPR_max − IPR_min).
    k : float
        Steepness of transition.
    log_gamma_c : float
        log(γ_c) — the threshold in log space.
    B : float
        Baseline (IPR at low γ).

    Returns
    -------
    ipr_predicted : ndarray
    """
    return A / (1.0 + np.exp(-k * (np.asarray(log_gamma) - log_gamma_c))) + B


def hill_model(gamma, n, gamma_c, ipr_min, ipr_max):
    """
    Hill-function alternative to sigmoid, for cases where sigmoid fails.

    IPR(γ) = ipr_min + (ipr_max − ipr_min) · γⁿ / (γ_c ⁿ + γⁿ)
    """
    gn = np.asarray(gamma) ** n
    return ipr_min + (ipr_max - ipr_min) * gn / (gamma_c ** n + gn)


# ── Entanglement / coherence measures (Week 11 supplement) ───────────────────

def von_neumann_entropy(rho_ss, n_chrom=N_CHROM):
    """
    Von Neumann entropy of the reduced density matrix (chromophore block only).

    S = −Tr(ρ_red · log ρ_red)

    Compare to IPR: both measure localisation, but entropy is basis-independent.
    """
    import qutip as qt
    full = rho_ss.full()
    rho_red = full[:n_chrom, :n_chrom]
    # Renormalise
    tr_red = np.real(np.trace(rho_red))
    if tr_red < 1e-12:
        return np.nan
    rho_red = rho_red / tr_red
    eigvals = np.linalg.eigvalsh(rho_red)
    eigvals = eigvals[eigvals > 1e-15]   # drop numerically negative eigenvalues
    return float(-np.sum(eigvals * np.log(eigvals)))


def coherence_l1(rho_ss, n_chrom=N_CHROM):
    """
    L1 norm of coherences (off-diagonal elements of the reduced density matrix).

    C_l1 = Σ_{i≠j} |ρ_ij|   (chromophore block only)
    """
    full    = rho_ss.full()[:n_chrom, :n_chrom]
    offdiag = full - np.diag(np.diag(full))
    return float(np.sum(np.abs(offdiag)))
