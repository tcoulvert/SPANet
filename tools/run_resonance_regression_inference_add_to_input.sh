#!/usr/bin/env bash
# Resonance regression inference: add predictions into the input HDF5.
# Set LOG_DIR and INPUT_H5, then run the command below (do not execute as-is without setting paths).

LOG_DIR="path/to/lightning_logs/version_0"
INPUT_H5="path/to/events.h5"

# Regression: adds INPUTS/RecoMass/reco_mass (and MASK) to INPUT_H5
python -m spanet.predict_resonance_regression "$LOG_DIR" \
  -tf "$INPUT_H5" \
  --add_prediction_to_input_h5 \
  --gpu

# Classification: adds INPUTS/MassCategory/probability (and MASK) to INPUT_H5
# python -m spanet.predict_resonance_regression "$LOG_DIR" \
#   -tf "$INPUT_H5" \
#   --classify \
#   --add_prediction_to_input_h5 \
#   --gpu

# Optional: -ckpt path/to/epoch_*.ckpt  -ef path/to/event.yaml  -bs 256  --fp16
