# GRIDEX training pipeline

This pipeline keeps GRIDEX code separate from MS-Swift. The recommended checkout layout is:

```text
workspace/
├── GRIDEX/
└── ms-swift/
```

The scripts may be launched from any directory. They resolve tools and plugins from `GRIDEX/training`; `MS_SWIFT_ROOT` only identifies the external MS-Swift checkout.

## 1. Install

The pipeline was developed against MS-Swift commit `d9010ed937680bfb23520ab1e9c2ad85f826be4a`. Pinning the revision avoids changes in CLI behavior.

```bash
git clone https://github.com/thinganhado/GRIDEX.git
git clone https://github.com/modelscope/ms-swift.git
git -C ms-swift checkout d9010ed937680bfb23520ab1e9c2ad85f826be4a

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./ms-swift
```

Install the CUDA, DeepSpeed, vLLM, and FlashInfer versions appropriate for the target cluster according to the pinned MS-Swift documentation. Training requires Linux, NVIDIA GPUs, and Bash; the scripts are not intended to execute in native Windows PowerShell.

If the repositories are not siblings, export the checkout path:

```bash
export MS_SWIFT_ROOT=/path/to/ms-swift
```

## 2. Required inputs

The portable defaults expect:

```text
GRIDEX/
├── dataset.csv
└── data/
    ├── metadata/region_diff_stats.csv
    ├── spectrograms/grid/<sample_id>_grid_img_edge_number_axes.png
    └── transcripts/mfa/<sample_id>.json
```

The factual CSV supplies rows by `(sample_id, region_id)`. It must be the reviewed `P=none` version whose only P values are `consonant`, `vowel`, and `none`. The currently tracked root `dataset.csv` is the legacy `unvoiced` copy; until it is replaced in a separate data update, pass `FACTUAL_CSV=/path/to/dataset_none.csv` when building Q2 data. The Q2 builder fails fast on legacy labels rather than silently mixing label schemes.

`region_diff_stats.csv` supplies `overlap_pixels`, which reconstructs the Query-1 ranking; CSV row order is never treated as rank. MFA JSON files supply timestamped word-tier transcripts. Every path can be overridden with the environment variables shown below.

The shared defaults are:

```bash
export DATA_ROOT=/path/to/GRIDEX/data
export OUTPUT_ROOT=/path/to/training_outputs
export CACHE_ROOT=/path/to/cache
```

## 3. Build Query-1 SFT data

The builder requires `ORDER_CSV`, creates both training and development JSON, verifies every constructed JSON-array target against descending `overlap_pixels`, and checks that every grid spectrogram exists. Missing scores, order mismatches, or missing images stop the build.

```bash
CSV_IN=/path/to/GRIDEX/dataset.csv \
ORDER_CSV=/path/to/region_diff_stats.csv \
IMAGE_DIR=/path/to/grid_spectrograms \
SPLIT_BY_PATH=1 \
STRICT_PATH_SPLIT=1 \
bash training/scripts/00_build_sft_query1_dataset.sh
```

The PDF protocol is the default: samples containing `_LA_T_` go to training and `_LA_D_` go to development. Unmatched samples are rejected, and the build requires the reported 10,320 training and 10,192 development samples. Do not use the random split for reported experiments because vocoder variants of a source utterance can cross partitions. For a deliberate non-paper subset experiment, override `EXPECTED_TRAIN_SAMPLES` and `EXPECTED_VAL_SAMPLES` explicitly.

## 4. SFT-1

```bash
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
bash training/scripts/01_stage1_sft_query1.sh
```

Defaults match the reported SFT-1 setup: bf16, LoRA rank/alpha 8/32, batch/accumulation 10/1, 3 epochs, learning rate `1e-4`, warmup ratio `0.03`, maximum length 768, and maximum pixels 524288. Validation loss is evaluated at save intervals.

SFT-1 uses an LLM-only LoRA (`freeze_llm=false`, `freeze_vit=true`, `freeze_aligner=true`) so the stage-specific adapter can learn the ordered JSON-list output while the vision backbone and aligner remain unchanged.

The optional multiturn experiment remains available as `01_stage1_sft_multiturn.sh`, but it is not part of the reported two-SFT curriculum.

## 5. GRPO-1 with the dedicated SFT-1 reference adapter

Use the same SFT-1 checkpoint to initialize the trainable policy and frozen reference model. Paths are supplied at runtime and are not hardcoded:

```bash
SFT1_ADAPTER=/path/to/sft1_query1/checkpoint-N

MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
POLICY_ADAPTERS="$SFT1_ADAPTER" \
REF_ADAPTERS="$SFT1_ADAPTER" \
bash training/scripts/02_stage2_grpo1.sh
```

`POLICY_ADAPTERS` becomes MS-Swift `--adapters`; `REF_ADAPTERS` becomes `--ref_adapters`. A fresh run requires both paths and requires them to identify the same SFT-1 checkpoint. If `MODEL_ID` is already an SFT-1 merged model, omit both variables and set `SFT1_MERGED_MODEL=1` to avoid applying SFT-1 twice.

GRPO-1 uses the train and validation files from step 3. Its defaults use 8 raw generations per prompt, bf16, LoRA 8/32, 2 epochs, learning rate `1e-5`, and prompt/completion limits 512/128. Evaluation is enabled with `--val_dataset`.

The Q1 JSON stores the ordered target in top-level `gt_regions` and `solution` fields as well as the SFT assistant turn. This prevents MS-Swift response preprocessing from removing the factual target needed by the GRPO reward. The reward refuses to train if neither explicit target is present.

The default is an unbiased raw-rollout distribution:

```text
GRPO_DEDUPE_ROLLOUTS=0
GRPO_DEDUPE_MAX_RETRIES=0
GRPO_RETRY_USE_GT_FEEDBACK=0
GRPO_RETRY_REQUIRE_IMPROVEMENT=0
```

Do not enable the legacy retry controls for the reported experiment; rejection and replacement condition the training distribution on reward-relevant properties.

The script automatically merges the selected GRPO-1 checkpoint by default. To merge explicitly:

```bash
ADAPTER_PATH=/path/to/grpo1/checkpoint-N \
OUTPUT_DIR=/path/to/grpo1_merged \
bash training/scripts/03_merge_grpo1_lora.sh
```

## 6. Build GT-ID and predicted-ID Query-2 data

First run Query-1 inference once per training sample and save every raw response without retrying, filtering on correctness, or selecting high-reward outputs. Then build both Q2 datasets:

```bash
P1_PRED_JSON=/path/to/all_raw_q1_predictions.json \
FACTUAL_CSV=/path/to/dataset_none.csv \
ORDER_CSV=/path/to/region_diff_stats.csv \
SPEC_ROOT=/path/to/grid_spectrograms \
MFA_ROOT=/path/to/mfa_json \
bash training/scripts/02b_build_grpo2_datasets.sh
```

For each valid predicted triplet `[p1, p2, p3]`, the builder looks up `(sample_id, region_id)` and writes factual tuples in exactly that order. A predicted ID is retained even when it is not in the GT top three. If a factual row is genuinely unavailable, `factual_mask` disables factual accuracy for only that slot; format and internal-consistency rewards remain active. Only malformed outputs that do not contain three distinct IDs in the valid grid range are excluded.

## 7. SFT-2 and merge

```bash
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
bash training/scripts/03_stage2_sft_query2.sh

ADAPTER_PATH=/path/to/sft2_query2/checkpoint-N \
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
bash training/scripts/03b_merge_sft2_lora.sh
```

SFT-2 is initialized independently from the same base model as SFT-1, matching the reported setup. Defaults are bf16, LoRA 32/64, batch/accumulation 4/8, 2 epochs, learning rate `5e-5`, warmup ratio `0.03`, maximum length 768, and maximum pixels 524288. Query 2 defines `P` as `{consonant, vowel, none}`; `none` is used when no phone is aligned.

## 8. GRPO-2 curriculum

Phase A trains on the unfiltered predicted-ID dataset. Phase B resumes the Phase-A GRPO checkpoint and trains on the GT-ID dataset.

```bash
BASE_MODEL_ID=/path/to/sft2_merged \
bash training/scripts/04_stage3_grpo2_curriculum.sh
```

The default is again raw rollout sampling: `GRPO_P2_RETRY_MISSING_FIELDS=0` and `GRPO_P2_MAX_RETRIES=0`. Phase A does not retry incomplete completions and does not filter predicted IDs based on Query-1 correctness.

To resume an interrupted GRPO-2 run, set `GRPO2_RESUME_CHECKPOINT` to a GRPO-2 `checkpoint-*` directory. An SFT-2 checkpoint must not be passed as a trainer resume checkpoint; merge SFT-2 first and supply it through `BASE_MODEL_ID`.

Merge the final adapter:

```bash
BASE_MODEL_ID=/path/to/sft2_merged \
ADAPTER_PATH=/path/to/grpo2_curriculum/checkpoint-N \
bash training/scripts/05_merge_grpo2_lora.sh
```

## Slurm

The files in `training/slurm` are cluster-neutral templates. Supply site-specific account and partition settings at submission time:

```bash
CONDA_ENV=gridex sbatch --account=<account> --partition=<partition> \
  training/slurm/stage2_grpo1_ms_swift.slurm
```

Environment overrides are exported by Slurm by default. The wrappers resolve `GRIDEX_ROOT` from their own location and do not `cd` into MS-Swift.
