# GRIDEX inference

End-to-end inference for GRIDEX, from raw audio to validated Q1 region IDs and
validated Q2 forensic explanations.

## Requirements

This Q1/Q2 package restores the completed pipeline from `gridex-xai` commit
`1c0646a`. The supported runtime is Linux/WSL with an NVIDIA GPU and Python
3.10 or 3.11. Install the Q1/Q2 model runtime with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The single `requirements.txt` contains the Qwen2.5-VL runtime. Raw-audio
preparation in `run.sh` also needs these packages in the same environment:

```bash
pip install faster-whisper==1.2.1 librosa==0.11.0 matplotlib==3.10.8 numpy==2.2.6
```

## GRIDEX pipeline

Supply audio, output, or trained-model paths to `run.sh` as environment variables:

```bash
AUDIO_DIR=/path/to/wav_or_flac_directory \
WORK_DIR=/path/to/output-directory \
Q1_MODEL_ID=/path/to/GRPO1-model \
Q2_MODEL_ID=/path/to/GRPO2-PhaseB-model \
Q1_BACKEND=transformers \
GPU_ID=0 \
bash run.sh
```

`Q1_BACKEND=transformers` selects the backend supported by the pinned
requirements. The optional vLLM backend requires installing vLLM separately.

Replace every `/path/to/...` placeholder before running:

- `AUDIO_DIR`: directory containing the uploaded `.wav` or `.flac` files.
- `WORK_DIR`: writable directory for generated images, transcripts, caches,
  Q1 predictions, and final Q2 results.

Download the final merged GRPO-1 and GRPO-2 Phase-B models from the
[GRIDEX model folder on Google Drive](https://drive.google.com/drive/folders/1jTE97u5S--HNFTnlZI1uIackRo3LNU0u?usp=sharing).
After downloading them, set `Q1_MODEL_ID` and `Q2_MODEL_ID` in the command
above to their respective extracted model directories.

The pipeline performs these stages:

1. Resample each audio file to 16 kHz and render the mel spectrogram.
2. Add the training 4x4 boundaries and IDs, then reproduce the training
   987x908 time/frequency-axes image.
3. Generate word timestamps with faster-whisper.
4. Write `q1_meta.json` and `q2_meta.csv` from the same sample inventory.
5. Run Q1 and collect `<sample_id>/json` predictions.
6. Join Q1 to Q2 strictly by `sample_id` and run Q2.
7. Write raw responses plus validated structured results.

For training-identical region labels, either place Arial at
`assets/arial.ttf` inside this repository or set
`GRIDEX_ARIAL_FONT=/path/to/arial.ttf`. Otherwise Pillow's available system
font fallback is used.

`run.sh` is the only shell entry point; it directly calls input
preparation, Q1 inference, and Q2 inference in that order.
