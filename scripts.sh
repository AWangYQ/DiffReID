#!/bin/bash
# Set DATASETS.ROOT_DIR / OUTPUT_DIR / MODEL.CLIP_PATH in the config file, or override them below.

# Train DiffReID on MSMT17 (stage 1: prompt tuning, stage 2: diffusion-based feature learning)
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  --denoise_layer 2 \
  SOLVER.SEED 1234 \
  DATASETS.NAMES 'msmt17' \
  MODEL.DDPM_STEPS 10 \
  MODEL.DDPM_FEATS 3 \
  MODEL.DDPM_QUERIES 16

# Test with a trained checkpoint
# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py \
#   --config_file configs/person/vit_clipreid.yml \
#   DATASETS.NAMES 'msmt17' \
#   TEST.WEIGHT 'your_checkpoint_path/ViT-B-16_60.pth'
