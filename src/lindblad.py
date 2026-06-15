"""
Core Lindblad master equation solver — 15-site source-sink model.

═══════════════════════════════════════════════════════════════════════
THE BUG FIX EXPLAINED
═══════════════════════════════════════════════════════════════════════
Pure dephasing with no source/sink ALWAYS reaches the maximally mixed
state (ρ_nn = 1/N for all n), giving IPR = 1/N = 0.0714 regardless of
γ.  This is not a code bug — it is a mathematical fact about Lindblad
dynamics in a closed system with uniform dephasing.

The fix is a 15-dimensional Hilbert space:
  • Sites 0–13 : the 14 LHCII chromophores
  • Site 14    : electronic ground state |g⟩

Three categories of collapse operators:
  1. Pure dephasing   L_n = √γ · |n⟩⟨n|            for n = 0..13
  2. Source (pump)    L_src = √κ_in · |src⟩⟨g|      (b601 ← ground)
  3. Sink (trap)      L_snk = √κ_out · |g⟩⟨snk|     (ground ← a610)

The source and sink create a non-equilibrium steady state where the
IPR depends non-trivially on γ, including the ENAQT sweet spot and the
Quantum Zeno localisation transition.
═══════════════════════════════════════════════════════════════════════

Unit convention
───────────────
All energies and rates in cm⁻¹ at the Python API level.
QuTiP receives values in rad/ps via CM1_TO_RADPS.
Time in mesolve is in ps.

  1 cm⁻¹ → 2π × c (cm/ps) = 2π × 2.998×10⁻² ≈ 0.18836 rad/ps
"""

import numpy as np
import qutip as qt
import warnings

# ── Unit conversion ──────────────────────────────────────────────────────────
C_CM_PER_PS   = 2.998e-2                   # speed of light (cm/ps)
CM1_TO_RADPS  = 2.0 * np.pi * C_CM_PER_PS  # ≈ 0.18836 rad ps⁻¹ per cm⁻¹

N_CHROM = 14   # chromophore sites
N_TOT   = 15   # total Hilbert space (chromophores + ground)
G_IDX   = 14   # index of ground state


# ── Hamiltonian embedding ─────────────────────────────────────────────────────

def embed_hamiltonian(H_14x14_cm1):
    """
    Embed 14×14 LHCII Hamiltonian (cm⁻¹) into 15×15 QuTiP Qobj (rad/ps).

    The ground state (row/col 14) has zero energy.
    """
    arr = np.zeros((N_TOT, N_TOT), dtype=complex)
    arr[:N_CHROM, :N_CHROM] = np.asarray(H_14x14_cm1, dtype=complex) * CM1_TO_RADPS
    H = qt.Qobj(arr)
    H.dims = [[N_TOT], [N_TOT]]
    return H


# ── Collapse operators ────────────────────────────────────────────────────────

def build_collapse_operators(gamma_cm1,
                              kappa_in_cm1,
                              kappa_out_cm1,
                              source_site=12,
                              sink_site=8):
    """
    Build all Lindblad collapse operators for the 15-site model.

    Parameters
    ----------
    gamma_cm1 : float
        Pure dephasing rate (cm⁻¹).  Sweep this to find the Zeno threshold.
    kappa_in_cm1 : float
        Incoherent pumping rate from ground into source_site (cm⁻¹).
    kappa_out_cm1 : float
        Irreversible trapping rate from sink_site to ground (cm⁻¹).
    source_site : int
        Chromophore index pumped by the light source (default 12 = b601).
    sink_site : int
        Chromophore index that traps excitation (default 8 = a610).

    Returns
    -------
    c_ops : list[Qobj]
        QuTiP collapse operators in units of √(rad/ps).
    """
    c_ops = []

    # ── 1. Pure dephasing: L_n = √γ · |n⟩⟨n| ──────────────────────────
    if gamma_cm1 > 0.0:
        sqrt_gamma = float(np.sqrt(gamma_cm1 * CM1_TO_RADPS))
        for n in range(N_CHROM):
            op = np.zeros((N_TOT, N_TOT), dtype=complex)
            op[n, n] = 1.0
            c_ops.append(sqrt_gamma * qt.Qobj(op))

    # ── 2. Source: |src⟩⟨g| — creates excitation at source_site ────────
    if kappa_in_cm1 > 0.0:
        sqrt_kin = float(np.sqrt(kappa_in_cm1 * CM1_TO_RADPS))
        op = np.zeros((N_TOT, N_TOT), dtype=complex)
        op[source_site, G_IDX] = 1.0    # |src⟩⟨g|
        c_ops.append(sqrt_kin * qt.Qobj(op))

    # ── 3. Sink: |g⟩⟨snk| — removes excitation at sink_site ────────────
    if kappa_out_cm1 > 0.0:
        sqrt_kout = float(np.sqrt(kappa_out_cm1 * CM1_TO_RADPS))
        op = np.zeros((N_TOT, N_TOT), dtype=complex)
        op[G_IDX, sink_site] = 1.0      # |g⟩⟨snk|
        c_ops.append(sqrt_kout * qt.Qobj(op))

    return c_ops


def ground_state_dm():
    """Initial density matrix: all population in electronic ground state."""
    psi0 = qt.basis(N_TOT, G_IDX)
    return qt.ket2dm(psi0)


# ── Steady-state solver ───────────────────────────────────────────────────────

def compute_steady_state(H, c_ops, method='direct', verbose=False):
    """
    Compute the non-equilibrium steady-state density matrix.

    Strategy:
      1. Try qt.steadystate() — algebraic solve, fast and exact.
      2. If that fails, fall back to time-propagation with mesolve.

    Parameters
    ----------
    H : Qobj
        15×15 Hamiltonian in rad/ps.
    c_ops : list[Qobj]
        Collapse operators in √(rad/ps).
    method : str
        'direct' (default) or 'iterative-gmres'.
    verbose : bool
        Print warnings on fallback.

    Returns
    -------
    rho_ss : Qobj or None
        Steady-state density matrix.  None if both methods fail.
    """
    # ── Algebraic steady state ──────────────────────────────────────────
    try:
        rho_ss = qt.steadystate(H, c_ops, method=method)
        _check_trace(rho_ss, verbose)
        return rho_ss
    except Exception as e:
        if verbose:
            print(f"  steadystate({method}) failed: {e}. Trying mesolve fallback…")

    # ── mesolve fallback ────────────────────────────────────────────────
    try:
        rho0   = ground_state_dm()
        # Run for ~20 × longest relaxation time.
        # With typical kappa_out ~ 100 cm^-1 = 18.8 rad/ps, τ_relax ~ 0.05 ps.
        t_max  = 20.0   # ps  — generous upper bound
        tlist  = np.linspace(0, t_max, 600)
        result = qt.mesolve(H, rho0, tlist, c_ops, [],
                            options=qt.Options(nsteps=20000, rtol=1e-8))
        rho_ss = result.states[-1]

        # Verify convergence: last two states agree to 0.1 %
        delta = (rho_ss - result.states[-2]).norm()
        if delta > 1e-3 and verbose:
            print(f"  WARNING: mesolve not fully converged (Δ={delta:.2e}). "
                  "Increase t_max in config.yaml.")
        _check_trace(rho_ss, verbose)
        return rho_ss
    except Exception as e:
        if verbose:
            print(f"  mesolve also failed: {e}")
        return None


def _check_trace(rho, verbose):
    tr = float(np.real(rho.tr()))
    if abs(tr - 1.0) > 1e-3 and verbose:
        print(f"  WARNING: trace = {tr:.6f} (expected 1.0). "
              "Check collapse operator construction.")


# ── Convenience wrapper for a single sweep point ──────────────────────────────

def run_one(H_14x14_cm1, gamma_cm1, kappa_in_cm1, kappa_out_cm1,
            source_site=12, sink_site=8,
            sigma_cm1=0.0, geometry_fn=None, rng=None):
    """
    Full pipeline for one (gamma, sigma, geometry) combination.

    Parameters
    ----------
    H_14x14_cm1 : ndarray (14,14)
        Base Hamiltonian in cm⁻¹ (before disorder).
    gamma_cm1 : float
        Dephasing rate for this simulation.
    kappa_in_cm1, kappa_out_cm1 : float
        Source and sink rates.
    source_site, sink_site : int
        Chromophore indices.
    sigma_cm1 : float
        Static disorder std (cm⁻¹); 0 = no disorder.
    geometry_fn : callable or None
        Function H_base → H_modified for geometry transformations.
    rng : np.random.Generator or None
        Pass for reproducibility.

    Returns
    -------
    rho_ss : Qobj or None
    """
    from src.hamiltonian import apply_disorder

    H = H_14x14_cm1.copy()
    if geometry_fn is not None:
        H = geometry_fn(H)
    if sigma_cm1 > 0.0:
        H = apply_disorder(H, sigma_cm1, rng=rng)

    H_qt  = embed_hamiltonian(H)
    c_ops = build_collapse_operators(gamma_cm1, kappa_in_cm1, kappa_out_cm1,
                                      source_site, sink_site)
    return compute_steady_state(H_qt, c_ops)
