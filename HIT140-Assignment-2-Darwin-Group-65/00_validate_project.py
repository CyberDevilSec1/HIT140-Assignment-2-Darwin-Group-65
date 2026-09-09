from pathlib import Path
import pandas as pd
import scipy
import matplotlib
import numpy

REQUIRED_FILES = [
    "data/fbref_2026_standard.csv",
    "data/fbref_2026_shooting.csv",
    "data/fbref_2026_goalkeeping.csv",
    "data/fbref_2026_misc.csv",
    "data/fifa_world_cup_2026_official_matches_104.csv",
]

print("=" * 70)
print("HIT140 FIFA WORLD CUP 2026 - PROJECT VALIDATION")
print("=" * 70)

missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    print("Missing files:")
    for f in missing:
        print(" -", f)
    raise SystemExit("\nPlease place the missing files in the data folder.")

print("All required data files were found.")
print("pandas:", pd.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("\nProject validation completed successfully.")