from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
from datetime import datetime
import numpy as np

from algorithms.ma_lfl import run_ma_lfl
from algorithms.ma_spi import run_ma_spi
from config import ExperimentConfig, load_config
from environments.gridworld import GridWorld, RewardFamily
from evaluation.metrics import correlation_pair
from evaluation.reporting import RewardEvaluationResult


# ----------------------------
# Arg parsing & overrides
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA-LfL reproduction pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override for base output directory (defaults to logging.base_dir in config).",
    )
    parser.add_argument(
        "--reward-families",
        type=str,
        nargs="+",
        default=None,
        choices=[member.value for member in RewardFamily],
        help="Reward families to evaluate. By default uses the value in the config file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional override for number of MA-SPI iterations.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for number of evaluation episodes per iteration.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a short smoke test by shrinking iterations and episodes.",
    )
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.iterations is not None:
        config.maspi.num_iterations = args.iterations
    if args.episodes is not None:
        config.maspi.evaluation_episodes_per_iteration = args.episodes
    if args.fast:
        config.maspi.num_iterations = min(3, config.maspi.num_iterations)
        config.maspi.evaluation_episodes_per_iteration = min(
            50, config.maspi.evaluation_episodes_per_iteration
        )
        config.maspi.episode_length = min(100, config.maspi.episode_length)
    # Lower bounds to avoid degenerate runs
    config.maspi.num_iterations = max(2, config.maspi.num_iterations)
    config.maspi.evaluation_episodes_per_iteration = max(1, config.maspi.evaluation_episodes_per_iteration)


# ----------------------------
# Logging
# ----------------------------

def setup_logging(base_output: Path) -> Path:
    log_dir = base_output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging initialized. Log file: %s", log_path)
    return log_path


# ----------------------------
# Utilities
# ----------------------------

def ground_truth_table(config: ExperimentConfig, reward_family: str, agent_id: int) -> np.ndarray:
    """Compute ground-truth reward table for a given reward family & agent."""
    env = GridWorld(
        size=config.environment.grid_size,
        start_positions=tuple(tuple(pos) for pos in config.environment.start_positions),
        goal_position=tuple(config.environment.goal_position),
        reward_family=RewardFamily(reward_family),
    )
    table = np.zeros((config.num_states, config.num_actions, config.num_actions), dtype=np.float32)
    for state_idx in range(config.num_states):
        joint_state = env.index_to_joint_state(state_idx)
        reward = env.reward(agent_id, joint_state)
        table[state_idx, :, :] = reward
    return table


def _log_legacy_evaluation_block(reward_family: str, evaluation: Optional[RewardEvaluationResult]) -> None:
    """Print the legacy single-view evaluation block (bare-only), kept for backward compatibility."""
    if evaluation is None:
        logging.warning("[Experiment:%s] Evaluation metrics unavailable.", reward_family)
        return

    logging.info(
        "[Experiment:%s] Agent 0 metrics - Pearson: %.4f | Spearman: %.4f",
        reward_family,
        evaluation.agent_metrics[0].pearson,
        evaluation.agent_metrics[0].spearman,
    )
    logging.info(
        "[Experiment:%s] Agent 1 metrics - Pearson: %.4f | Spearman: %.4f",
        reward_family,
        evaluation.agent_metrics[1].pearson,
        evaluation.agent_metrics[1].spearman,
    )
    if evaluation.trend_stages:
        logging.info(
            "[Experiment:%s] Trend stages: %s",
            reward_family,
            ", ".join(str(s) for s in evaluation.trend_stages),
        )
        logging.info(
            "[Experiment:%s] Trend Pearson: %s",
            reward_family,
            ", ".join(f"{value:.4f}" for value in evaluation.trend_pearson),
        )
        logging.info(
            "[Experiment:%s] Trend Spearman: %s",
            reward_family,
            ", ".join(f"{value:.4f}" for value in evaluation.trend_spearman),
        )
        # success flags like before
        success_flags = []
        for agent_idx, metrics in enumerate(evaluation.agent_metrics):
            is_positive = metrics.pearson >= 0.4 and metrics.spearman >= 0.4
            success_flags.append(is_positive)
            logging.info(
                "[Experiment:%s] Agent %d correlation >=0.4 ? %s",
                reward_family,
                agent_idx,
                "YES" if is_positive else "NO",
            )
        if evaluation.trend_stages:
            increasing_trend = all(
                earlier <= later for earlier, later in zip(evaluation.trend_pearson, evaluation.trend_pearson[1:])
            ) and all(
                earlier <= later for earlier, later in zip(evaluation.trend_spearman, evaluation.trend_spearman[1:])
            )
            logging.info(
                "[Experiment:%s] Correlation trend non-decreasing? %s",
                reward_family,
                "YES" if increasing_trend else "NO",
            )


def _maybe_load_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning("Failed to read %s: %s", path, e)
    return None


def _log_dual_metrics_if_available(eval_dir: Path, reward_family: str) -> Optional[Dict[str, dict]]:
    """
    If evaluation/metrics_bare.json and evaluation/metrics_shaped.json exist,
    print both to log and return dict for summary.
    """
    bare_path = eval_dir / "metrics_bare.json"
    shaped_path = eval_dir / "metrics_shaped.json"
    bare = _maybe_load_json(bare_path)
    shaped = _maybe_load_json(shaped_path)

    if bare is None and shaped is None:
        return None

    if bare is not None:
        mean = bare.get("mean_metrics", {})
        logging.info("[Experiment:%s][bare] Pearson (mean): %.4f | Spearman (mean): %.4f",
                     reward_family, mean.get("pearson", float("nan")), mean.get("spearman", float("nan")))
        for i, m in enumerate(bare.get("agent_metrics", [])):
            logging.info("[Experiment:%s][bare] Agent %d - Pearson: %.4f | Spearman: %.4f",
                         reward_family, i, m.get("pearson", float("nan")), m.get("spearman", float("nan")))
        if bare.get("trend_stages"):
            logging.info("[Experiment:%s][bare] Trend stages: %s",
                         reward_family, ", ".join(str(s) for s in bare["trend_stages"]))
            logging.info("[Experiment:%s][bare] Trend Pearson: %s",
                         reward_family, ", ".join(f"{v:.4f}" for v in bare.get("trend_pearson", [])))
            logging.info("[Experiment:%s][bare] Trend Spearman: %s",
                         reward_family, ", ".join(f"{v:.4f}" for v in bare.get("trend_spearman", [])))

    if shaped is not None:
        mean = shaped.get("mean_metrics", {})
        logging.info("[Experiment:%s][shaped] Pearson (mean): %.4f | Spearman (mean): %.4f",
                     reward_family, mean.get("pearson", float("nan")), mean.get("spearman", float("nan")))
        for i, m in enumerate(shaped.get("agent_metrics", [])):
            logging.info("[Experiment:%s][shaped] Agent %d - Pearson: %.4f | Spearman: %.4f",
                         reward_family, i, m.get("pearson", float("nan")), m.get("spearman", float("nan")))
        if shaped.get("trend_stages"):
            logging.info("[Experiment:%s][shaped] Trend stages: %s",
                         reward_family, ", ".join(str(s) for s in shaped["trend_stages"]))
            logging.info("[Experiment:%s][shaped] Trend Pearson: %s",
                         reward_family, ", ".join(f"{v:.4f}" for v in shaped.get("trend_pearson", [])))
            logging.info("[Experiment:%s][shaped] Trend Spearman: %s",
                         reward_family, ", ".join(f"{v:.4f}" for v in shaped.get("trend_spearman", [])))

    return {"bare": bare, "shaped": shaped}


# ----------------------------
# Experiment orchestration
# ----------------------------

def run_single_experiment(config: ExperimentConfig, reward_family: str, output_dir: Path) -> Dict:
    # Fix reward family for this run
    config.environment.reward_family = reward_family

    # 1) MA-SPI
    logging.info("[Experiment:%s] Starting MA-SPI...", reward_family)
    maspi_artifacts = run_ma_spi(config)
    maspi_output = output_dir / "ma_spi"
    maspi_artifacts.save(maspi_output)
    logging.info("[Experiment:%s] MA-SPI completed. Artifacts saved to %s", reward_family, maspi_output)

    # 2) MA-LfL (this function internally does evaluation with the current reporting.py)
    stage_datasets = [stage.dataset for stage in maspi_artifacts.stages]
    malfl_output = output_dir / "ma_lfl"
    logging.info("[Experiment:%s] Starting MA-LfL...", reward_family)
    malfl_artifacts = run_ma_lfl(config, stage_datasets, malfl_output)
    logging.info("[Experiment:%s] MA-LfL completed. Artifacts saved to %s", reward_family, malfl_output)

    # 3) Legacy single-view evaluation block (kept)
    evaluation: Optional[RewardEvaluationResult] = malfl_artifacts.evaluation
    _log_legacy_evaluation_block(reward_family, evaluation)

    # 4) NEW: If dual-mode JSONs exist (bare/shaped), log them too (backward compatible)
    eval_dir = malfl_output / "evaluation"
    dual_metrics = _log_dual_metrics_if_available(eval_dir, reward_family)

    # 5) Build result payload (kept compatible with your summary writer)
    predicted_tables = [table.detach().cpu().numpy() for table in malfl_artifacts.reward_learning.predicted_reward_tables]
    reward_paths = [
        str((malfl_output / "reward_learning" / f"agent_{idx}" / "rewards.pt").relative_to(output_dir))
        for idx in range(len(predicted_tables))
    ]

    # Legacy evaluation dict
    legacy_eval_dict = {
        "agents": [
            {"pearson": m.pearson, "spearman": m.spearman}
            for m in (evaluation.agent_metrics if evaluation else [])
        ],
        "mean": {
            "pearson": evaluation.mean_metrics.pearson if evaluation else None,
            "spearman": evaluation.mean_metrics.spearman if evaluation else None,
        },
        "trend_stages": evaluation.trend_stages if evaluation else [],
        "trend_pearson": evaluation.trend_pearson if evaluation else [],
        "trend_spearman": evaluation.trend_spearman if evaluation else [],
    }

    # Optionally attach dual metrics to result (won't affect summary serialization below)
    result = {
        "reward_family": reward_family,
        "evaluation": legacy_eval_dict,
        "evaluation_modes": dual_metrics,  # may be None if files not present
        "predicted_rewards": predicted_tables,
        "reward_artifacts": reward_paths,
    }
    return result


def compute_cross_correlation(
    config: ExperimentConfig,
    experiments: Dict[str, Dict],
) -> List[Dict[str, object]]:
    reward_families = list(experiments.keys())
    results: List[Dict[str, object]] = []
    for pred_family, data in experiments.items():
        predicted_tables = data["predicted_rewards"]
        for target_family in reward_families:
            for agent_id in range(2):
                true_table = ground_truth_table(config, target_family, agent_id)
                pcc, scc = correlation_pair(predicted_tables[agent_id], true_table)
                results.append(
                    {
                        "predicted_family": pred_family,
                        "target_family": target_family,
                        "agent_id": agent_id,
                        "pearson": pcc,
                        "spearman": scc,
                    }
                )
    return results


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    reward_families = args.reward_families or [config.environment.reward_family]
    base_output = Path(args.output_dir) if args.output_dir else Path(config.logging.base_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(base_output)

    logging.info("Configuration file: %s", args.config)
    logging.info(
        "Reward families to process: %s",
        ", ".join(reward_families),
    )
    logging.info(
        "MA-SPI iterations=%d, episodes/stage=%d, episode_length=%d",
        config.maspi.num_iterations,
        config.maspi.evaluation_episodes_per_iteration,
        config.maspi.episode_length,
    )
    logging.info(
        "MA-LfL reward epochs=%d, reward batch size=%d",
        config.malfl.reward_epochs,
        config.malfl.reward_batch_size,
    )
    logging.info("Log file located at %s", log_path)

    experiments: Dict[str, Dict] = {}
    for reward_family in reward_families:
        family_dir = base_output / reward_family
        family_dir.mkdir(exist_ok=True)
        logging.info("Running experiment for reward family: %s", reward_family)
        result = run_single_experiment(config, reward_family, family_dir)
        experiments[reward_family] = result

    # Write summary.json (keep your original convention: strip large arrays, keep shapes)
    summary_path = base_output / "summary.json"
    serializable = {}
    for family, data in experiments.items():
        serializable_family = dict(data)
        # Write shapes only
        serializable_family["predicted_rewards_shape"] = [
            list(arr.shape) for arr in data["predicted_rewards"]
        ]
        serializable_family["predicted_rewards"] = None
        serializable[family] = serializable_family
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    # Cross-correlation only if multiple families
    if len(experiments) > 1:
        cross_results = compute_cross_correlation(config, experiments)
        cross_path = base_output / "cross_correlation.csv"
        header = "predicted_family,target_family,agent_id,pearson,spearman"
        with cross_path.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            for row in cross_results:
                handle.write(
                    f"{row['predicted_family']},{row['target_family']},{row['agent_id']},"
                    f"{row['pearson']:.6f},{row['spearman']:.6f}\n"
                )
        logging.info("Saved cross-correlation table to %s", cross_path)

    logging.info("Summary saved to %s", summary_path)
    logging.info("Run complete.")


if __name__ == "__main__":
    main()
