"""Data generation runner for T1 (GridWorld + MA-SPI) and T2 (MPE + PPO)."""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from data_gen.adapters import LearningPhaseData, ma_spi_to_learning_phases, ppo_to_learning_phases
from data_gen.ma_spi.gridworld_sampler import run_ma_spi_phases
from data_gen.custom_ppo.mpe_runner import generate_mpe_ppo_data
from data_gen.custom_ppo.multiwalker_runner import generate_multiwalker_ppo_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data for Project9")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_phases(phases: list[LearningPhaseData], output_dir: Path, seed: int, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"{prefix}_seed{seed}_{ts}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(phases, f)
    return out_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    env_name = config.get("env") or config.get("env_name", "")

    print(f"[gen_data] Config: {config_path} (env={env_name})")
    print(f"[gen_data] Seed: {args.seed}")

    if env_name.lower() == "gridworld":
        raw_phases = run_ma_spi_phases(config, seed=args.seed)
        phases = ma_spi_to_learning_phases(
            [
                {
                    "phase_idx": p.phase_idx,
                    "policy_logits": p.policy_logits,
                    "theta_sequence": p.theta_sequence,
                    "trajectories": p.trajectories,
                }
                for p in raw_phases
            ]
        )
        out_dir = Path("outputs") / "data" / "t1"
        out_path = save_phases(phases, out_dir, seed=args.seed, prefix="t1_ma_spi")
        print(f"[gen_data] Generated {len(phases)} phases -> {out_path}")
        for i, phase in enumerate(phases[:3]):
            num_traj = len(phase.trajectories)
            param_shape = phase.policy_params.shape
            print(f"  Phase {i}: trajectories={num_traj}, policy_params_shape={param_shape}")
    elif env_name.lower() in {"mpe_simple_spread", "simple_spread", "t2"}:
        raw_checkpoints = generate_mpe_ppo_data(config, seed=args.seed, save_dir=Path("outputs") / "raw" / "t2")
        phases = ppo_to_learning_phases(raw_checkpoints)
        out_dir = Path("outputs") / "data" / "t2"
        out_path = save_phases(phases, out_dir, seed=args.seed, prefix="t2_ppo")
        print(f"[gen_data] Generated {len(phases)} PPO phases -> {out_path}")
        for i, phase in enumerate(phases[:3]):
            agents = list(phase.policy_params.keys()) if isinstance(phase.policy_params, dict) else []
            traj_count = len(phase.trajectories)
            theta_shape = next(iter(phase.policy_params.values())).shape if agents else "unknown"
            print(f"  Phase {i}: agents={agents}, theta_shape={theta_shape}, rollouts={traj_count}")
    elif env_name.lower() in {"multiwalker", "t3"}:
        raw_checkpoints = generate_multiwalker_ppo_data(config, seed=args.seed, save_dir=Path("outputs") / "raw" / "t3")
        phases = ppo_to_learning_phases(raw_checkpoints)
        out_dir = Path("outputs") / "data" / "t3"
        out_path = save_phases(phases, out_dir, seed=args.seed, prefix="t3_ppo")
        print(f"[gen_data] Generated {len(phases)} PPO phases -> {out_path}")
        for i, phase in enumerate(phases[:3]):
            agents = list(phase.policy_params.keys()) if isinstance(phase.policy_params, dict) else []
            traj_count = len(phase.trajectories)
            theta_shape = next(iter(phase.policy_params.values())).shape if agents else "unknown"
            print(f"  Phase {i}: agents={agents}, theta_shape={theta_shape}, rollouts={traj_count}")
    else:
        raise SystemExit(f"Unsupported env '{env_name}' for Phase 1 generation")


if __name__ == "__main__":
    main()
