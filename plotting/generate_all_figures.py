"""Generate all fully scripted manuscript figures in numerical order."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent

for script in (
    "figure01_binning_schematic.py",
    "figure02_binning_pseudocode.py",
    "figure03_model_cross_section.py",
    "figure03_site_legend.py",
    "figure04_process_uncertainty.py",
    "figure05_first_total_effect.py",
    "figure06_second_order.py",
    "figure07_third_fourth_order.py",
    "figure08_order_contributions.py",
    "figure09_process_rankings.py",
):
    subprocess.run([sys.executable, str(HERE / script)], check=True)

