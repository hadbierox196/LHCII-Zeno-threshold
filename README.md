# LHCII Zeno Threshold Project

**Research question:** Does the ratio γ/Δε² determine Quantum Zeno regime
entry in LHCII chromophore networks, and does γ_c predict exciton
localisation topology in a geometry-dependent manner?

**OSF pre-registration DOI:** `https://doi.org/10.17605/OSF.IO/SXTHP`

---

## ⚡ Quick start in Google Colab

```python
# ── Cell 1: mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/lhcii-zeno-threshold')

# ── Cell 2: install dependencies (run once per Colab session)
!pip install qutip numpy scipy matplotlib h5py pandas \
             joblib biopython pyyaml tqdm seaborn scikit-learn -q

# ── Cell 3: run any week
%run scripts/01_dimer_validation.py
%run scripts/02_build_hamiltonian.py   # fill in Müh 2010 values first!
%run scripts/03_ipr_validation.py
%run scripts/05_parameter_sweep.py     # the FIXED sweep (Week 5)
%run scripts/06_threshold_extraction.py
%run scripts/07_spatial_localization.py
%run scripts/08_robustness_controls.py
%run scripts/09_figure_finalization.py

# ── Or run the full Snakemake pipeline
!pip install snakemake -q
!snakemake --cores 1
```

---

## The Week 5 bug and its fix

**Bug:** Pure dephasing with no source or sink always produces the
maximally mixed state (IPR = 1/14 = 0.0714 for all γ). The sweep
measured nothing physical.

**Fix:** A 15-dimensional Hilbert space (14 chromophores + ground state)
with an incoherent pump (source at b601) and an irreversible trap
(sink at a610). This produces a non-equilibrium steady state where IPR
depends non-trivially on γ.

See `src/lindblad.py` for the detailed explanation and implementation.

---

## Project structure

```
lhcii-zeno-threshold/
├── README.md
├── requirements.txt          ← Python dependencies
├── config.yaml               ← ALL tunable parameters
├── Snakefile                 ← reproducible pipeline
│
├── src/                      ← core physics modules
│   ├── hamiltonian.py        ← load / build LHCII Hamiltonian
│   ├── lindblad.py           ← 15-site source-sink Lindblad model
│   ├── metrics.py            ← IPR, PR, entropy, sigmoid model
│   └── visualization.py      ← publication-quality plotting
│
├── scripts/                  ← one script per week
│   ├── 01_dimer_validation.py
│   ├── 02_build_hamiltonian.py   ← ⚠ fill in Müh 2010 values
│   ├── 03_ipr_validation.py
│   ├── 05_parameter_sweep.py     ← ★ FIXED (source + sink)
│   ├── 06_threshold_extraction.py
│   ├── 07_spatial_localization.py
│   ├── 08_robustness_controls.py
│   └── 09_figure_finalization.py
│
├── data/
│   └── hamiltonians/         ← place 1RWT.pdb here
│
├── results/                  ← generated outputs (not committed)
└── figures/                  ← generated figures (not committed)
```

---

## Critical first step: build the Hamiltonian (Week 2)

Open `scripts/02_build_hamiltonian.py` and replace all values marked
`← REPLACE` with the exact numbers from:

> Müh, F. et al. (2010) α-helices direct excitation energy flow in
> the Fenna-Matthews-Olson protein. *PNAS* 107, 16297–16302.
> DOI: 10.1073/pnas.1004206107

Then run:
```python
%run scripts/02_build_hamiltonian.py
```

This saves `results/lhcii_hamiltonian_mueh2010.h5`.  Every downstream
script loads from this file — no values are hardcoded elsewhere.

---

## Unit conventions

| Quantity | Python API | QuTiP internal |
|---|---|---|
| Site energies | cm⁻¹ | rad/ps (× 0.18836) |
| Coupling J | cm⁻¹ | rad/ps |
| Dephasing γ | cm⁻¹ | rad/ps |
| Source κ_in | cm⁻¹ | rad/ps |
| Sink κ_out | cm⁻¹ | rad/ps |
| Time (mesolve) | ps | — |

Conversion: `1 cm⁻¹ × 2π × c(cm/ps) = 1 cm⁻¹ × 0.18836 rad/ps`

---

## Download PDB file (for Week 7)

```bash
# In Colab:
!mkdir -p data
!wget -q https://files.rcsb.org/download/1RWT.pdb -O data/1RWT.pdb
```

---

## Config parameters (`config.yaml`)

All sweep parameters, source/sink rates, and figure settings are in
`config.yaml`.  Change them there; no script edits needed.

Key parameters:
- `source_sink.kappa_in_cm1`: pumping rate into b601 (default 10 cm⁻¹)
- `source_sink.kappa_out_cm1`: trapping rate at a610 (default 100 cm⁻¹)
- `sweep.gamma_min_cm1 / gamma_max_cm1`: sweep range
- `sweep.n_realizations`: disorder realisations per (γ, σ) point

---

## Checklist before submitting (Week 12)

- [ ] OSF pre-registration timestamped before Week 5 sweep
- [ ] `sweep_raw.h5` uploaded to OSF (primary dataset)
- [ ] `lhcii_hamiltonian_mueh2010.h5` verified against Müh 2010
- [ ] All figures: PNG (300 dpi) + SVG
- [ ] `snakemake --forceall --cores 1` runs clean from scratch
- [ ] GitHub repo public with this README and OSF DOI badge

---

## Citation

If you use this code, please cite:

> Farooq, H. (2025). *Quantum Zeno threshold mapping in LHCII chromophore
> networks.* OSF pre-registration DOI: [YOUR DOI].
> GitHub: https://github.com/[YOUR_USERNAME]/lhcii-zeno-threshold
