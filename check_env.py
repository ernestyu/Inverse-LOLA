"""Phase 0 environment and import sanity check."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


LIBRARIES = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pettingzoo",
    "gymnasium",
    "matplotlib",
    "scipy",
    "cma",
    "omegaconf",
]

PROJECT_MODULES = [
    "envs.gridworld",
    "envs.mpe_simple_spread",
    "envs.multiwalker",
    "learners.ma_spi.gridworld_ma_spi",
    "learners.ppo.mpe_runner",
    "learners.ppo.multiwalker_runner",
    "data_gen.ma_spi.gridworld_sampler",
    "data_gen.custom_ppo.mpe_runner",
    "data_gen.custom_ppo.multiwalker_runner",
    "data_gen.adapters",
    "inverse.ma_lfl.core",
    "inverse.ilola.stage_a_independent",
    "inverse.ilola.stage_b_cmaes",
    "algorithms.baseline_lfl",
    "algorithms.i_logel",
    "algorithms.i_lola",
    "models.reward_nets",
    "models.feature_maps",
    "models.dynamics",
    "evaluation.metrics",
    "evaluation.plots",
    "evaluation.report",
    "evaluation.induced_train",
    "runners.gen_data",
    "runners.run_ma_lfl",
    "runners.run_ilogel",
    "runners.run_ilola",
    "runners.run_induced",
]


def check_libraries() -> None:
    print("== Library versions ==")
    failures: list[str] = []
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"- {name}: {version}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            failures.append(name)
            print(f"- {name}: ERROR ({exc})")
    if failures:
        raise SystemExit(f"Missing/failed libraries: {', '.join(failures)}")


def check_project_imports() -> None:
    print("== Project module imports ==")
    failures: list[str] = []
    for module_name in PROJECT_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"- {module_name}: OK")
        except Exception as exc:  # pragma: no cover - diagnostic only
            failures.append(module_name)
            print(f"- {module_name}: ERROR ({exc})")
    if failures:
        raise SystemExit(f"Modules failed to import: {', '.join(failures)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    print(f"Repository root: {repo_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    check_libraries()
    check_project_imports()
    print("Phase 0 OK")


if __name__ == "__main__":
    main()
