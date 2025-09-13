#!/bin/bash

# Usage:
#   ./run.sh [epochs] [model_type] [mode]
#
# Examples:
#   ./run.sh                     # default train with config defaults
#   ./run.sh 50 lstm             # override epochs & model
#   ./run.sh "" tcn sweep        # Optuna sweep for TCN
#   ./run.sh "" lstm predict     # Run prediction using pretrained LSTM

EPOCHS=${1:-}
MODEL=${2:-}
MODE=${3:-train}  # default to train

echo "Running pipeline in mode: $MODE"

# Detect package manager and set up command prefix
if command -v uv &> /dev/null && [ -f "pyproject.toml" ]; then
    echo "Using uv for dependency management"
    PYTHON_CMD="uv run python"
elif [ -f ".venv/bin/python" ]; then
    echo "Using local virtual environment"
    PYTHON_CMD=".venv/bin/python"
elif command -v python &> /dev/null; then
    echo "Using system Python"
    PYTHON_CMD="python"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

# Initialize command
if [[ "$MODE" == "predict" ]]; then
    CMD="$PYTHON_CMD -m src.predict"
else
    CMD="$PYTHON_CMD -m src.pipeline"
fi

# Inject model config if specified
[ -n "$MODEL" ] && CMD="$CMD model_cfg=$MODEL"

if [[ "$MODE" == "sweep" ]]; then
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: You must specify a model for sweep mode (e.g. lstm, tcn, deepar)"
        exit 1
    fi
    CMD="$CMD optuna_${MODEL}"
elif [[ "$MODE" == "train" ]]; then
    # Override epochs only in training mode
    [ -n "$EPOCHS" ] && CMD="$CMD model_cfg.epochs=$EPOCHS"
fi

echo "Command: $CMD"
eval $CMD
