# LHCII Quantum Zeno Threshold Mapping

[![OSF Pre-registration](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://doi.org/10.17605/OSF.IO/SXTHP)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)
[![DOI](https://zenodo.org/badge/1269738652.svg)](https://doi.org/10.5281/zenodo.20749253)

**Author:** Hassan Farooq, Sargodha Medical College  
**OSF Pre-registration:** https://doi.org/10.17605/OSF.IO/SXTHP  
**Status:** Manuscript in preparation

---

## Research Question

In the LHCII light-harvesting chromophore network, at what pure dephasing rate γ does
the system transition from environment-assisted quantum transport (ENAQT) into the
Quantum Zeno localisation regime — and does this threshold depend on chromophore
network geometry?

---

## Key Findings

| Quantity | Value |
|---|---|
| ENAQT optimum γ_ENAQT (symmetric, σ=0) | **112.5 cm⁻¹** |
| Zeno threshold γ_c (symmetric, σ=0) | **701.2 cm⁻¹** (95% CI: 692.4–710.1) |
| Physiological dephasing range at 300 K | 50–200 cm⁻¹ |
| Geometry dependence of γ_c | 1.02× fold (not significant) |
| LHCII regime in vivo | **ENAQT optimum** — γ_phys ≈ γ_ENAQT |

**Headline result:** The ENAQT optimum (γ_ENAQT = 112.5 cm⁻¹, dephasing
timescale ≈ 47 fs) falls within the physiological protein conformational fluctuation
range at 300 K. The Quantum Zeno threshold is 6.2-fold above physiological
conditions. LHCII appears to be tuned to operate at its quantum transport optimum
in vivo.

---

## Method

- **Model:** 15-dimensional Lindblad master equation (14 chromophores + ground state)
- **Boundary conditions:** Incoherent source at Chl *b* 601; irreversible sink at Chl *a* 610
- **Hamiltonian:** Müh et al. 2010 *J. Phys. Chem. B* (QM/MM, 14-site LHCII monomer)
- **Metric:** Inverse Participation Ratio (IPR) of the non-equilibrium steady-state density matrix
- **Solver:** Pure NumPy/SciPy — scipy.linalg.solve on the 225×225 Liouville superoperator
- **Sweep:** 40 γ × 3 σ × 2 geometry × 100 realisations = **24,000 simulations in 3.9 min**

---

## The Core Physics Fix

Pure dephasing with no source or sink always drives the system to the maximally mixed
state: IPR = 1/N = 0.0714 for all γ. This is mathematically guaranteed — not a code
bug — and means a naive sweep measures nothing physical.

The fix is a **15-dimensional Hilbert space** (14 chromophores + electronic ground
state |g⟩) with three categories of collapse operator:

| Operator | Form | Physics |
|---|---|---|
| Dephasing | L_n = √γ \|n⟩⟨n\| for n = 0..13 | Site-energy fluctuations from protein bath |
| Source | L_src = √κ_in \|b601⟩⟨g\| | Incoherent photon absorption at Chl b 601 |
| Sink | L_snk = √κ_out \|g⟩⟨a610\| | Irreversible trapping toward reaction centre |

This creates a non-equilibrium steady state where IPR depends non-trivially on γ.
See `src/lindblad.py` for the full derivation and implementation.

---

## Repository Structure

```
LHCII-Zeno-threshold/
│
├── src/
│   ├── hamiltonian.py        # LHCII Hamiltonian loader and builder
│   ├── lindblad.py           # 15-site source-sink Lindblad model (physics fix)
│   ├── metrics.py            # IPR, participation ratio, sigmoid model
│   └── visualization.py      # Publication-quality plotting utilities
│
├── scripts/
│   ├── 00_quick_diagnostic.py    # Run this FIRST — validates source-sink fix
│   ├── 01_dimer_validation.py    # 2-site dimer Lindblad limiting cases
│   ├── 02_build_hamiltonian.py   # Build LHCII Hamiltonian from Müh 2010
│   ├── 03_ipr_validation.py      # IPR formula validation and 5-point preview
│   ├── 05_parameter_sweep.py     # Full γ sweep (optimised NumPy/SciPy solver)
│   ├── 06_threshold_extraction.py # Sigmoid fit and γ_c extraction
│   ├── 07_spatial_localization.py # Exciton density maps on PDB coordinates
│   ├── 08_robustness_controls.py  # Parameter sensitivity analysis
│   └── 09_figure_finalization.py  # Manuscript figures + Results section draft
│
├── data/
│   └── hamiltonians/         # Source documentation; place 1RWT.pdb here
│
├── config.yaml               # All tunable parameters
├── Snakefile                 # Full reproducible pipeline
├── COLAB_SETUP.py            # Paste into Colab Cell 1 to configure session
└── requirements.txt          # Python dependencies
```

---

## Quick Start (Google Colab)

**Cell 1 — Session setup**
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/LHCII-Zeno-threshold')
!pip install qutip==4.7.3 numpy scipy matplotlib h5py pandas tqdm pyyaml -q
```

**Cell 2 — Keepalive thread (prevents disconnection during sweep)**
```python
import threading, time, sys
def _keep():
    for i in range(9999):
        time.sleep(55)
        sys.stdout.write(f'\r keepalive {i+1}'); sys.stdout.flush()
threading.Thread(target=_keep, daemon=True).start()
```

**Cell 3 — Validate the source-sink model (30 seconds)**
```python
%run scripts/00_quick_diagnostic.py
```

**Cell 4 — Run the full sweep (~4 minutes)**
```python
%run scripts/05_parameter_sweep.py
```

**Cell 5 — Extract γ_c and generate figures**
```python
%run scripts/06_threshold_extraction.py
%run scripts/09_figure_finalization.py
```

---

## Hamiltonian Setup (Required Before First Run)

Open `scripts/02_build_hamiltonian.py` and replace all values marked
`← REPLACE` with exact numbers from:

> Müh, F., Madjet, M. E., & Renger, T. (2010). Structure-based identification
> of energy sinks in plant light-harvesting complex II.
> *J. Phys. Chem. B*, **114**, 13517–13535.
> DOI: [10.1021/jp106323e](https://doi.org/10.1021/jp106323e)

Then run the script once to generate `results/lhcii_hamiltonian_mueh2010.h5`.
Every downstream script loads from this file.

**Site ordering (0-indexed, Müh 2010):**

| Index | Site | Energy (cm⁻¹) | Role |
|---|---|---|---|
| 0–7 | Chl a 602–609 | 15,020–15,220 | Antenna |
| 8–11 | Chl a 610–613 | 14,730–15,020 | Low-energy cluster |
| **12** | **Chl b 601** | **15,880** | **Source site** |
| 13 | Chl b 606 | 15,970 | Antenna |

---

## Reproducing All Results

From a clean Colab session with the repo on Google Drive:

```python
%run COLAB_SETUP.py                    # install + navigate
%run scripts/02_build_hamiltonian.py   # requires Müh 2010 values
%run scripts/05_parameter_sweep.py     # ~4 min, saves sweep_raw_v2.h5
%run scripts/06_threshold_extraction.py
%run scripts/09_figure_finalization.py
```

Or with Snakemake (local environment):

```bash
pip install snakemake
snakemake --cores 1 --forceall
```

---

## Scope Reduction Note

The pre-registered sweep (OSF) specified 80 γ × 5 σ × 2 geom × 200 realisations
= 160,000 simulations. This was reduced to 40 × 3 × 2 × 100 = **24,000 simulations**
due to Google Colab compute constraints (Pivot Trigger 3, documented in OSF
pre-registration before execution). The NumPy/SciPy solver completed the reduced
sweep in 3.9 minutes on Colab CPU. Qualitative conclusions are unchanged.

---

## Data Availability

| Dataset | Location |
|---|---|
| Primary sweep (`sweep_raw_v2.h5`) | OSF: https://doi.org/10.17605/OSF.IO/SXTHP |
| LHCII Hamiltonian (`lhcii_hamiltonian_mueh2010.h5`) | OSF (same DOI) |
| Pre-registration (timestamped) | OSF (same DOI) |
| Analysis code | This repository |

Large binary files (HDF5, PDB) are not committed to this repository.

---

## Unit Conventions

| Quantity | API unit | Internal unit |
|---|---|---|
| Site energies, coupling J | cm⁻¹ | rad ps⁻¹ (× 0.1884) |
| Dephasing rate γ | cm⁻¹ | rad ps⁻¹ |
| Source κ_in, sink κ_out | cm⁻¹ | rad ps⁻¹ |

Conversion: `1 cm⁻¹ = 2π × 2.998 × 10⁻² rad ps⁻¹ ≈ 0.1884 rad ps⁻¹`

---

## Citation

If you use this code or data, please cite:

```
Farooq, H. (2025). Quantum Zeno threshold mapping in LHCII chromophore networks:
physiological dephasing rates coincide with the ENAQT optimum.
Manuscript in preparation.
OSF pre-registration: https://doi.org/10.17605/OSF.IO/SXTHP
GitHub: https://github.com/hadbierox196/LHCII-Zeno-threshold
```

---

## License

MIT License. See `LICENSE` for details.
