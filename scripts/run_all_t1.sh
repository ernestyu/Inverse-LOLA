#!/usr/bin/env bash
set -euo pipefail

# Placeholder pipeline for task 1 (GridWorld)
make gen CONF=configs/t1_gridworld.yaml SEED=${SEED:-0}
