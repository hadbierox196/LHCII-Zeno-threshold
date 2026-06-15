"""
Snakemake pipeline — LHCII Zeno Threshold Project

Run entire pipeline:
  snakemake --cores 1

Dry run (check DAG without executing):
  snakemake --cores 1 --dry-run

Generate DAG image:
  snakemake --dag | dot -Tpng > figures/pipeline_dag.png

Individual rules:
  snakemake results/week1_dimer_validation.done --cores 1
  snakemake results/lhcii_hamiltonian_mueh2010.h5 --cores 1
  snakemake results/week3_ipr_5points.csv --cores 1
  snakemake results/sweep_raw.h5 --cores 1
  snakemake results/week6_gamma_c_results.csv --cores 1
  snakemake results/week7_localization_site_table.csv --cores 1
  snakemake results/week8_sensitivity_table.csv --cores 1
  snakemake figures/fig1_ipr_vs_gamma.png --cores 1
"""

configfile: "config.yaml"


# ── Target: everything ────────────────────────────────────────────────────────
rule all:
    input:
        "results/week1_dimer_validation.done",
        "results/lhcii_hamiltonian_mueh2010.h5",
        "results/week3_ipr_5points.csv",
        "results/sweep_raw.h5",
        "results/week6_gamma_c_results.csv",
        "results/week7_localization_site_table.csv",
        "results/week8_sensitivity_table.csv",
        "figures/fig1_ipr_vs_gamma.png",
        "figures/fig2_gamma_c_vs_sigma.png",
        "figures/fig3_spatial_localization.png",
        "figures/fig4_robustness.png",


# ── Week 1 ────────────────────────────────────────────────────────────────────
rule validate_dimer:
    output:
        touch("results/week1_dimer_validation.done")
    log:
        "logs/week1_dimer_validation.log"
    shell:
        "python scripts/01_dimer_validation.py > {log} 2>&1"


# ── Week 2 ────────────────────────────────────────────────────────────────────
rule build_hamiltonian:
    output:
        "results/lhcii_hamiltonian_mueh2010.h5"
    log:
        "logs/week2_hamiltonian.log"
    shell:
        "python scripts/02_build_hamiltonian.py > {log} 2>&1"


# ── Week 3 ────────────────────────────────────────────────────────────────────
rule ipr_validation:
    input:
        "results/lhcii_hamiltonian_mueh2010.h5"
    output:
        "results/week3_ipr_5points.csv"
    log:
        "logs/week3_ipr_validation.log"
    shell:
        "python scripts/03_ipr_validation.py > {log} 2>&1"


# ── Week 5 (FIXED sweep) ─────────────────────────────────────────────────────
rule parameter_sweep:
    input:
        "results/lhcii_hamiltonian_mueh2010.h5"
    output:
        "results/sweep_raw.h5"
    log:
        "logs/week5_parameter_sweep.log"
    shell:
        "python scripts/05_parameter_sweep.py > {log} 2>&1"


# ── Week 6 ────────────────────────────────────────────────────────────────────
rule threshold_extraction:
    input:
        "results/sweep_raw.h5"
    output:
        "results/week6_gamma_c_results.csv"
    log:
        "logs/week6_threshold_extraction.log"
    shell:
        "python scripts/06_threshold_extraction.py > {log} 2>&1"


# ── Week 7 ────────────────────────────────────────────────────────────────────
rule spatial_localization:
    input:
        "results/lhcii_hamiltonian_mueh2010.h5",
        "results/week6_gamma_c_results.csv"
    output:
        "results/week7_localization_site_table.csv"
    log:
        "logs/week7_spatial_localization.log"
    shell:
        "python scripts/07_spatial_localization.py > {log} 2>&1"


# ── Week 8 ────────────────────────────────────────────────────────────────────
rule robustness_controls:
    input:
        "results/sweep_raw.h5",
        "results/week6_gamma_c_results.csv"
    output:
        "results/week8_sensitivity_table.csv"
    log:
        "logs/week8_robustness.log"
    shell:
        "python scripts/08_robustness_controls.py > {log} 2>&1"


# ── Week 9 ────────────────────────────────────────────────────────────────────
rule finalize_figures:
    input:
        "results/sweep_raw.h5",
        "results/week6_gamma_c_results.csv",
        "results/week7_localization_site_table.csv",
        "results/week8_sensitivity_table.csv"
    output:
        "figures/fig1_ipr_vs_gamma.png",
        "figures/fig2_gamma_c_vs_sigma.png",
        "figures/fig3_spatial_localization.png",
        "figures/fig4_robustness.png",
    log:
        "logs/week9_figure_finalization.log"
    shell:
        "python scripts/09_figure_finalization.py > {log} 2>&1"


# ── Clean ─────────────────────────────────────────────────────────────────────
rule clean:
    shell:
        "rm -rf results/ figures/ logs/ __pycache__ src/__pycache__ scripts/__pycache__"
