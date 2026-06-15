# Hamiltonian data sources

## Primary: Müh et al. 2010

**Reference:** Müh, F. et al. (2010). α-helices direct excitation energy flow
in the Fenna-Matthews-Olson protein. *PNAS* **107**, 16297–16302.
DOI: [10.1073/pnas.1004206107](https://doi.org/10.1073/pnas.1004206107)

**Site energies:** Table 1 (QM/MM computed Qy transition energies, cm⁻¹)

**Coupling constants:** Table 2 or Supplementary Table (all J_mn values, cm⁻¹)

**Site ordering used in this project** (0-indexed):
```
0: Chl a 602    1: Chl a 603    2: Chl a 604    3: Chl a 605
4: Chl a 606    5: Chl a 607    6: Chl a 608    7: Chl a 609
8: Chl a 610    9: Chl a 611   10: Chl a 612   11: Chl a 613
12: Chl b 601  13: Chl b 606
```

After entering all values in `scripts/02_build_hamiltonian.py`, run that
script to generate `results/lhcii_hamiltonian_mueh2010.h5`.

---

## Robustness control: Novoderezhkin et al. 2011

**Reference:** Novoderezhkin, V. I. et al. (2011). Mixing of exciton and
charge-transfer states in Photosystem II reaction centers: modeling of Stark
spectra with modified Redfield theory. *Phys. Chem. Chem. Phys.* **13**.

Extract their Table 1 site energies and Table 2 coupling constants, enter
them into a copy of `scripts/02_build_hamiltonian.py`, and save to
`results/lhcii_hamiltonian_novo2011.h5`.

---

## PDB file

Place the LHCII crystal structure here:

```bash
wget -q https://files.rcsb.org/download/1RWT.pdb -O data/1RWT.pdb
# or for the higher-resolution structure:
wget -q https://files.rcsb.org/download/2BHW.pdb -O data/2BHW.pdb
```

The Week 7 script reads Mg atom coordinates from PDB 1RWT.
