"""Run the sensitivity calculations in dependency order."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent

for script in (
    "process_sensitivity_common.py",
    "process_second_order.py",
    "process_third_fourth_order.py",
):
    subprocess.run([sys.executable, str(HERE / script)], check=True)

