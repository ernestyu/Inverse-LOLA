#!/usr/bin/env bash
set -euo pipefail

# Placeholder pipeline for task 2 (MPE simple spread)
make gen CONF=configs/t2_mpe_simple_spread.yaml SEED=${SEED:-0}
