# MA-LfL Reproduction

This repository implements the full reproduction pipeline for the paper **"Multi-Agent Learning from Learners"**. The code follows the engineering blueprint in docs/engineering_guide.md and reproduces the MA-SPI + MA-LfL baseline, including policy estimation, reward recovery, correlation metrics, and diagnostic plots.

## Environment Setup

1. Create a Python 3.10+ virtual environment and activate it.
2. Install the Python dependencies:

   `ash
   pip install -r requirements.txt
   `

If you plan to use a CUDA-enabled GPU, adjust the --extra-index-url in equirements.txt to match your CUDA version before installing PyTorch.

## Configuration

Hyperparameters and logging options are stored in config.yaml. Key sections:

- environment: grid size, start/goal positions, and initial reward family.
- maspi: MA-SPI loop settings (iterations, episode length, update counts, random seed).
- malfl: policy estimation, target computation, and joint reward/potential optimisation hyperparameters.
- optimization: learning rates for the MA-SPI actors/critics.
- logging: base output directory and data retention flags.

Defaults follow Table 3 of the paper. Adjust the YAML if you want to shorten runs or explore different regimes.

## Running Experiments

Use main.py to execute the full pipeline:

`ash
python main.py --config config.yaml --reward-families homogeneous heterogeneous
`

Important CLI options:

- --output-dir: override the base output directory.
- --reward-families: run one or multiple reward families (homogeneous, heterogeneous).
- --iterations / --episodes: override MA-SPI iteration or episode counts for experimentation.
- --fast: shrink iterations/episodes for a quick smoke test (e.g. CI runs).

### Smoke Test

To validate the installation without the full workload:

`ash
python main.py --fast --reward-families homogeneous
`

This completes in a few minutes and verifies that MA-SPI, MA-LfL, and evaluation artifacts are produced.

### Full Reproduction

For a faithful reproduction of the paper results, run both reward families:

`ash
python main.py --reward-families homogeneous heterogeneous
`

This generates:

- ma_spi/ — on-policy datasets D_h, policy logits, and training losses.
- ma_lfl/ — policy estimation artifacts, reward targets, recovered reward networks, and evaluation plots.
- evaluation/ — correlation metrics, reward trend curves (Figure 3 analogue), and heatmaps comparable to the Appendix figures.
- summary.json — aggregate metrics per experiment.
- cross_correlation.csv — Table 2 style sanity-check showing the highest correlation on the correct pairing of reward families.

Random seeds are fixed to ensure deterministic runs. Delete the corresponding output sub-directory before rerunning to avoid mixing results.

## Project Structure

`
├── algorithms/
│   ├── ma_spi.py        # On-policy data generation (Algorithm 1)
│   └── ma_lfl.py        # Reward inference, policy estimation, metrics (Algorithm 2)
├── agents/
│   └── ma_spi_agent.py  # Soft policy iteration agent with shared α temperature
├── environments/
│   └── gridworld.py     # Deterministic 3×3 grid world with M_hom / M_het rewards
├── models/              # Policy, Q, reward, and potential networks
├── data/structures.py   # On-policy trajectory containers (D_h)
├── evaluation/          # Correlation metrics and plotting helpers
├── main.py              # End-to-end orchestration and reporting
├── config.yaml          # Default hyperparameters
└── outputs/             # Created at runtime with reproducible artifacts
`

## Notes

- All computations share a single global temperature α, enforced through ExperimentConfig and propagated to every module.
- The code saves complete intermediate artifacts (policies, trajectories, reward networks) to satisfy the reproducibility checklist in docs/engineering_guide.md.
- Cross-correlation diagnostics and trend plots provide immediate feedback on reward recovery quality, mirroring Table 1/Table 2 and Figure 3 from the paper.

Refer to docs/engineering_guide.md for the detailed engineering plan that guided this implementation.
