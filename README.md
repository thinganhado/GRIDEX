# GRIDEX

Official implementation of **GRIDEX: Region-Grounded Forensic Explanations for Vocoder-Based Synthetic Speech**.

## 1. Preprocessing

### 1.1 Obtain the audio dataset

GRIDEX uses vocoded spoof audio and its paired bona fide audio. The dataset source is the [Voc.v4](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/09-asvspoof-vocoded-trn).

### 1.2 Generate training assets

Install preprocessing dependencies:

```bash
pip install -r training/preprocessing-requirements.txt
```

Run the training asset preparation script:

```bash
PAIR_CSV=/path/to/real_fake_pairs.csv \
DATA_ROOT="$PWD/data" \
bash training/scripts/prepare_assets.sh
```

It creates the following files:

```text
data/metadata/region_diff_stats.csv
data/spectrograms/grid/<sample_id>_grid_img_edge_number_axes.png
data/transcripts/mfa/<sample_id>.json
```

## 2. Training

Training uses [MS-Swift](https://github.com/modelscope/ms-swift) at commit `d9010ed937680bfb23520ab1e9c2ad85f826be4a`. Keep MS-Swift as a separate checkout. From the GRIDEX repository root, install the pinned revision and set its path:

```bash
git clone https://github.com/modelscope/ms-swift.git ../ms-swift
git -C ../ms-swift checkout d9010ed937680bfb23520ab1e9c2ad85f826be4a
pip install -e ../ms-swift
export MS_SWIFT_ROOT="$PWD/../ms-swift"
```

### 2.1 Build the Q1 training format

```bash
CSV_IN="$PWD/dataset.csv" \
ORDER_CSV="$PWD/data/metadata/region_diff_stats.csv" \
IMAGE_DIR="$PWD/data/spectrograms/grid" \
bash training/scripts/build_sft_q1_data.sh
```

### 2.2 Train Q1

Run SFT-1:

```bash
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
bash training/scripts/01_sft_q1.sh
```

Run GRPO-1:

```bash
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
POLICY_ADAPTERS=/path/to/sft1/checkpoint-N \
REF_ADAPTERS=/path/to/sft1/checkpoint-N \
bash training/scripts/02_grpo_q1.sh
```

### 2.3 Build the Q2 training format

Run the merged GRPO-1 model and save predictions:

```bash
python inference/q1.py \
  --backend transformers \
  --model-id /path/to/merged-grpo1-model \
  --meta-json "$PWD/data/processed/stage1_query1_train_swift.json" \
  --output-dir "$PWD/data/processed/q1_raw_predictions" \
  --output-jsonl "$PWD/data/processed/q1_raw_predictions.jsonl"
```

Build GT-ID data and the predicted-ID data:

```bash
FACTUAL_CSV=/path/to/reviewed_dataset_none.csv \
P1_PRED_JSON="$PWD/data/processed/q1_raw_predictions.jsonl" \
ORDER_CSV="$PWD/data/metadata/region_diff_stats.csv" \
SPEC_ROOT="$PWD/data/spectrograms/grid" \
MFA_ROOT="$PWD/data/transcripts/mfa" \
bash training/scripts/build_grpo_q2_data.sh
```

### 2.4 Train Q2

Run SFT-2 and the two GRPO-2 curriculum phases:

```bash
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
bash training/scripts/03_sft_q2.sh

BASE_MODEL_ID=/path/to/sft2_merged \
bash training/scripts/04_grpo_q2.sh
```

## 3. Inference

```bash
Q1_BACKEND=transformers \
AUDIO_DIR=/path/to/wav_or_flac_directory \
WORK_DIR=/path/to/inference-output \
Q1_MODEL_ID=/path/to/merged-grpo1-model \
Q2_MODEL_ID=/path/to/merged-grpo2-model \
GPU_ID=0 \
bash inference/run.sh
```
