Language: **English** [简体中文](cn_FRONTEND.md)

This release supports `preprocess.py`, `main_reflow.py`, `train_frontend_vc_gan.py`, and `train_index.py` with the SoftVcStylizer frontend, the full audio discriminator, NSF-HifiGAN, and the upstream Hugging Face ContentVec encoder. BigVGAN, earlier stylizers, Fairseq, and experimental residual models are excluded. No personalized checkpoints, speaker embeddings, datasets, or indexes are distributed.

Use Python 3.11 on Windows and install the matching CUDA runtime before the project requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CUDA command follows the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). Torch and Torchaudio should both report `2.11.0+cu126`; `torch.cuda.is_available()` should return `True`. `requirements-pitch.txt` is optional for Parselmouth, WORLD, CREPE, or FCPE; the default RMVPE path does not need those packages. GUI and live audio-device workflows are outside this dependency manifest.

Supply these public pretrained resources locally. They are ignored by Git and are not copied from another experiment by the migration:

| Resource | Default path | Notes |
| --- | --- | --- |
| Hugging Face ContentVec | `pretrain/contentvec/pytorch_model.bin` | Use the upstream-converted checkpoint, not the Fairseq legacy `.pt`. |
|  [Whisper large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) | `pretrain/ppg-large/large-v2.pt` | Standard checkpoint with `dims` and `model_state_dict`; the PPG loader keeps the first three quarters of the audio encoder. Text decoding is not required. |
| HuBERT Soft | `pretrain/hubert/hubert-soft.pt` | Public HuBERT Soft weights matching the vendored implementation. |
| RMVPE | `pretrain/rmvpe/model.pt` | Default pitch extractor. |
| NSF-HifiGAN | `vocoder.ckpt` from YAML | Keep its matching `config.json` beside the checkpoint. Its sample rate and hop must match the data configuration. |
| ECAPA speaker encoder | `speechbrain/spkrec-ecapa-voxceleb` | Downloaded by SpeechBrain when frontend preprocessing is explicitly run; cached under `frontend.speaker_encoder_cache`. This setting can instead point at a local model directory. |

Start with `configs/frontend-vc-gan.yaml`. Set training/validation audio paths, the vocoder checkpoint, and the base experiment directory. The base DDSP model must have matching sample rate, block size, encoder name, and feature dimensions. The existing upstream `train_reflow.py` can train that base model; the frontend trainer extracts and freezes only its DDSP component.

The frontend resource settings are:

```yaml
frontend:
  enabled: true
  base_checkpoint: null          # defaults to env.expdir; or a checkpoint/directory
  checkpoint: null               # required for frontend inference
  speaker_embedding: null        # optional inference override; defaults to the training speaker mean
  index_dir: exp/feature_index   # generic output/input directory
  contentvec_index: null         # optional explicit file overriding index_dir
  hubert_index: null
  whisper_index: null
  whisper_ckpt: pretrain/ppg-large/large-v2.pt
  hubert_ckpt: pretrain/hubert/hubert-soft.pt
  model: {}                     # SoftVcStylizer constructor options; saved with checkpoints
```

An explicitly supplied CLI resource path overrides YAML. Frontend preprocessing computes each speaker's arithmetic mean ECAPA embedding from successfully processed training clips and saves the same FP32 means in both `data.train_path/speaker_means/<speaker_id>.npy` and `data.valid_path/speaker_means/<speaker_id>.npy`. Validation clips do not contribute to the means. Training loads each speaker's saved mean once, and validation reuses the training means. This works with cached and uncached features; no global mean file or similarity threshold is used. With the frontend enabled, `main_reflow.py` automatically loads `data.train_path/speaker_means/<spk_id>.npy` using `-id` / `--spk_id` (default 1), once before processing the audio. The `-c` config's `data.train_path` takes precedence over the model's saved training path. The optional inference-only `speaker_embedding` setting or `--speaker_embedding` overrides this lookup with an explicit mean file or directory; a directory also uses `<spk_id>.npy`. The trainer resumes the latest numbered checkpoint in `env.style_dir` by default; `frontend.checkpoint` or `--frontend_checkpoint` selects a specific GAN checkpoint while new outputs still go to `env.style_dir`.

Run the workflow from this repository directory:

```powershell
# Populate data/train/audio and data/val/audio, then extract all frontend features.
.\.venv\Scripts\python.exe preprocess.py -c configs/frontend-vc-gan.yaml --store_dtype fp16 -j 1

# Train the frontend against your base checkpoint. Resume uses env.style_dir.
.\.venv\Scripts\python.exe train_frontend_vc_gan.py -c configs/frontend-vc-gan.yaml --base_checkpoint exp/base/model_10000.pt

# Build ContentVec, HuBERT, and Whisper indexes from the configured train/val paths.
.\.venv\Scripts\python.exe train_index.py -c configs/frontend-vc-gan.yaml --output_dir exp/feature_index

# Convert using explicit base/frontend checkpoints; load the speaker mean from data.train_path.
.\.venv\Scripts\python.exe main_reflow.py -m exp/base/model_10000.pt -i input.wav -o output.wav -c configs/frontend-vc-gan.yaml --frontend_checkpoint exp/frontend/model_500.pt -id 1 --index_dir exp/feature_index -fr 1
```

`-fr 0` disables retrieval. As in the source, a positive `-fr` enables distribution mapping; its magnitude is not an interpolation weight. Retrieval retains the existing standard-deviation formula, including its behavior for constant channels. Individual index paths can be passed with `--contentvec_index`, `--hubert_index`, and `--whisper_index`. Index files retain the original per-speaker dictionary format and filenames. `--speaker_id` and `--speaker_subdir` on `train_index.py` control the dictionary key and input feature subdirectory for a speaker.

For ordinary upstream inference without a frontend, omit the frontend settings or pass `--no_frontend`. The original `configs/reflow.yaml` remains the upstream baseline. Preprocessing defaults to two workers for the upstream path and one for the frontend path because every worker owns all three encoders. `--workers` overrides that choice. FP16 storage applies to unit/mel feature streams, while F0, volume, and speaker embeddings stay FP32. Data loading restores FP32 features when `cache_fp16` is false, including validation.

Two intentional corrections are included: waveform crops follow the configured output block size, and ECAPA inputs are separately resampled to 16 kHz on the selected device. The original training-rate waveform is unchanged. Reprocess the dataset for this release; old cached features, embeddings, and indexes are not a compatibility target.

Run regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures and temporary checkpoints. Public pretrained-model smoke checks exercise actual encoder/vocoder computation when the resources are present; they do not establish trained voice quality.
