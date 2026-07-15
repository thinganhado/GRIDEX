# GRIDEX

Official implementation of **GRIDEX: Grid-Grounded Forensic Explanations for Deepfake Spectrogram Analysis**.

The repository contains the project-owned dataset and training pipeline. MS-Swift remains a separate dependency; no upstream MS-Swift files are copied into this repository.

## Repository layout

- `dataset.csv`: region-level factual annotations (`sample_id`, `region_id`, `T`, `F`, `P`, `En`)
- `training/scripts`: SFT-1, GRPO-1, SFT-2, and GRPO-2 entry points
- `training/tools`: dataset construction utilities
- `training/plugins`: GRIDEX reward functions for MS-Swift
- `training/slurm`: portable Slurm wrapper templates

See [training/README.md](training/README.md) for setup, required inputs, and the complete training sequence.
