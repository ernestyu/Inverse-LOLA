#!/usr/bin/env bash
set -euo pipefail

# Placeholder pipeline for task 3 (Multiwalker)
make gen CONF=configs/t3_multiwalker.yaml SEED=${SEED:-0}
