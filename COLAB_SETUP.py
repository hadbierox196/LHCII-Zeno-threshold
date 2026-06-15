"""
COLAB SESSION SETUP
═══════════════════
Paste this entire file into ONE Colab cell and run it.
It installs packages, mounts Drive, and navigates to your project.

After this cell completes, use  %run scripts/XX_...py  for each week.
"""

# ── 1. Install all dependencies ───────────────────────────────────────────────
import subprocess, sys

packages = [
    'qutip',
    'numpy',
    'scipy',
    'matplotlib',
    'h5py',
    'pandas',
    'joblib',
    'biopython',
    'pyyaml',
    'tqdm',
    'seaborn',
    'scikit-learn',
]

print('Installing packages…')
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet'] + packages,
    check=True
)
print('✓ Packages installed.')

# ── 2. Mount Google Drive ─────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_AVAILABLE = True
    print('✓ Drive mounted at /content/drive')
except ImportError:
    DRIVE_AVAILABLE = False
    print('(Not in Colab — skipping Drive mount)')

# ── 3. Navigate to project directory ─────────────────────────────────────────
import os

# EDIT THIS PATH to match where you uploaded / cloned the repo on Drive:
PROJECT_PATH = '/content/drive/MyDrive/lhcii-zeno-threshold'

if DRIVE_AVAILABLE and os.path.exists(PROJECT_PATH):
    os.chdir(PROJECT_PATH)
    print(f'✓ Working directory: {os.getcwd()}')
elif not DRIVE_AVAILABLE:
    # Running locally — assume already in project root
    print(f'Working directory: {os.getcwd()}')
else:
    print(f'⚠  Project path not found: {PROJECT_PATH}')
    print('   Options:')
    print('   A) Upload the lhcii-zeno-threshold folder to your Google Drive')
    print('      at MyDrive/lhcii-zeno-threshold/ and re-run this cell.')
    print('   B) Clone from GitHub:')
    print('      !git clone https://github.com/[YOUR_USERNAME]/lhcii-zeno-threshold.git')
    print('      Then update PROJECT_PATH above.')

# ── 4. Verify project structure ───────────────────────────────────────────────
expected = ['src/lindblad.py', 'src/hamiltonian.py', 'src/metrics.py',
            'scripts/05_parameter_sweep.py', 'config.yaml']
missing = [p for p in expected if not os.path.exists(p)]

if missing:
    print(f'\n⚠  Missing files: {missing}')
    print('   Make sure the full repo is in the project directory.')
else:
    print('✓ Project structure verified.')

# ── 5. Create output directories ──────────────────────────────────────────────
for d in ['results', 'figures', 'logs', 'data']:
    os.makedirs(d, exist_ok=True)
print('✓ Output directories ready.')

# ── 6. Print run order ────────────────────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════════
Setup complete.  Run weeks in this order:

  %run scripts/00_quick_diagnostic.py    ← START HERE (30 sec)

  %run scripts/01_dimer_validation.py    Week 1
  %run scripts/02_build_hamiltonian.py   Week 2  ← fill Müh 2010 values first!
  %run scripts/03_ipr_validation.py      Week 3
  %run scripts/05_parameter_sweep.py     Week 5  ← THE FIXED SWEEP
  %run scripts/06_threshold_extraction.py Week 6
  %run scripts/07_spatial_localization.py Week 7
  %run scripts/08_robustness_controls.py  Week 8
  %run scripts/09_figure_finalization.py  Week 9

Or run the full pipeline:
  !snakemake --cores 1

═══════════════════════════════════════════════════════════════
""")
