"""Run MA-LfL evaluation on T1 GridWorld data."""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cfg = "configs/t1_gridworld.yaml"
    seed = 0
    try:
        subprocess.run(["make", "lfl", f"CONF={cfg}", f"SEED={seed}"], check=True)
    except FileNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "runners.run_ma_lfl", "--config", cfg, "--seed", str(seed)],
            check=True,
        )


if __name__ == "__main__":
    main()
