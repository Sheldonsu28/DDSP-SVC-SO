Language: **English** [简体中文](./cn_README.md)

# DDSP-SVC

## 0. Introduction

DDSP-SVC is an open source singing voice conversion project. This branch builds on upstream DDSP / Rectified-Flow and provides training and inference workflows with the SoftVcStylizer frontend, the full audio discriminator, and NSF-HifiGAN.

Compared with upstream DDSP, this repository improves pronunciation in synthesized speech and more faithfully reconstructs the target speaker's timbre, at the cost of higher training resource requirements.

This branch supports the following four entry points:

| Entry point | Purpose |
| --- | --- |
| [`preprocess.py`](preprocess.py) | Extract ContentVec, Whisper PPG, HuBERT Soft, speaker embeddings, pitch, volume, and mel spectrogram features. |
| [`train_frontend_vc_gan.py`](train_frontend_vc_gan.py) | Freeze the base model's DDSP parameters and train the SoftVcStylizer frontend with the full audio discriminator. |
| [`train_index.py`](train_index.py) | Build separate FAISS feature indexes for ContentVec, HuBERT, and Whisper. |
| [`main_reflow.py`](main_reflow.py) | Convert audio using the base model, the optional SoftVcStylizer frontend, and feature retrieval. |

ContentVec uses this codebase's Hugging Face implementation and the `pytorch_model.bin` weight format. 

The release does not include personal trained weights, speaker embeddings, datasets, preprocessing caches, or retrieval indexes. Users must prepare these resources or generate them through preprocessing and training. See [cn_FRONTEND.md](cn_FRONTEND.md) for the Chinese frontend guide, [FRONTEND.md](FRONTEND.md) for the English guide.

Disclaimer: Please make sure to only train DDSP-SVC models with **legally obtained authorized data**, and do not use these models and any audio they synthesize for illegal purposes. The author of this repository is not responsible for any infringement, fraud and other illegal acts caused by the use of these model checkpoints and audio.

## 1. Installing the dependencies

We recommend first installing PyTorch from the [official website](https://pytorch.org/), then run:

```bash
pip install -r requirements.txt
```

python 3.11 (windows) + cuda 13.0 + torch 2.9.1 + torchaudio 2.9.1 works.

## 2. Configuring the pretrained model

- Feature Encoder (choose only one):

(1) Download the pre-trained [ContentVec](https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin?download=true) encoder and put it under `pretrain/contentvec` folder.

(2) Download the pre-trained [HubertSoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) encoder and put it under `pretrain/hubert` folder, and then modify the configuration file at the same time.

- Vocoder:

Download and unzip the pre-trained [NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip) vocoder 

or use the https://github.com/openvpi/SingingVocoders project to fine-tune the vocoder for higher sound quality.

Then rename the checkpoint file and place it at the location specified by the 'vocoder.ckpt' parameter in the configuration file. The default value is `pretrain/nsf_hifigan/model`.

The 'config.json' of the vocoder needs to be at the same directory, for example, `pretrain/nsf_hifigan/config.json`.

- Pitch extractor:

Download the pre-trained [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) extractor and unzip it into `pretrain/` folder.

## 3. Preprocessing

Put all the training dataset (.wav format audio clips) in the below directory: `data/train/audio`. Put all the validation dataset (.wav format audio clips) in the below directory: `data/val/audio`. You can also run

```bash
python draw.py
```

to help you select validation data (you can adjust the parameters in `draw.py` to modify the number of extracted files and other parameters)

Then run the preprocessor:

```bash
python preprocess.py -c configs/reflow.yaml -j <number of processes>
```

NOTE 1: The default configuration is suitable for training a 44.1kHz high sampling rate synthesizer with an RTX-4060 graphics card.

NOTE 2: Please keep the sampling rate of all audio clips consistent with the sampling rate in the yaml configuration file ! If it is not consistent, the program can be executed safely, but the resampling during the preprocessing will be very slow.

NOTE 3: The total number of the audio clips for training dataset is recommended to be about 1000, especially long audio clip can be cut into short segments, which will speed up the training, but the duration of all audio clips should not be less than 2 seconds. If there are too many audio clips, you need a large internal-memory or set the 'cache_all_data' option to false in the configuration file.

NOTE 4: The total number of the audio clips for validation dataset is recommended to be about 10, please don't put too many or it will be very slow to do the validation.

NOTE 5: If your dataset is not very high quality, set 'f0_extractor' to 'rmvpe' in the config file.

NOTE 6: Multi-speaker training is supported now. The 'n_spk' parameter in configuration file controls whether it is a multi-speaker model. If you want to train a **multi-speaker** model, audio folders need to be named with **positive integers not greater than 'n_spk'** to represent speaker ids, the directory structure is like below:

```bash
# training dataset
# the 1st speaker
data/train/audio/1/aaa.wav
data/train/audio/1/bbb.wav
...
# the 2nd speaker
data/train/audio/2/ccc.wav
data/train/audio/2/ddd.wav
...

# validation dataset
# the 1st speaker
data/val/audio/1/eee.wav
data/val/audio/1/fff.wav
...
# the 2nd speaker
data/val/audio/2/ggg.wav
data/val/audio/2/hhh.wav
...
```

If 'n_spk' = 1, the directory structure of the **single speaker** model is still supported, which is like below:

```bash
# training dataset
data/train/audio/aaa.wav
data/train/audio/bbb.wav
...
# validation dataset
data/val/audio/ccc.wav
data/val/audio/ddd.wav
...
```

## 4. Training

```bash
python train_reflow.py -c configs/reflow.yaml
```

After training starts, a weight is temporarily saved every 'interval_val' step, and a weight is permanently saved every 'interval_force_save' step. These two configuration items can be modified according to the situation.

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## 5. Visualization

```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```

Test audio samples will be visible in TensorBoard after the first validation.


## 6. Non-real-time VC

```bash
python main_reflow.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -step <infer_step> -method <method> -ts <t_start>
```

'infer_step' is the number of sampling steps for rectified-flow ODE, 'method' is 'euler' or 'rk4', 't_start' is the start time point of ODE, which needs to be larger than or equal to t_start in the configuration file, it is recommended to keep it equal (the default is 0.0).

You can use "-mix" option to design your own vocal timbre, below is an example:

```bash
# Mix the timbre of 1st and 2nd speaker in a 0.5 to 0.5 ratio
python main_reflow.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}"
```

Other options about the f0 extractor and response threshold, see:

```bash
python main_reflow.py -h
```

## 7. Real-time VC

Start a simple GUI with the following command:

```bash
python gui_reflow.py
```

The front-end uses technologies such as sliding window, cross-fading, SOLA-based splicing and contextual semantic reference, which can achieve sound quality close to non-real-time synthesis with low latency and resource occupation.

## 8. Acknowledgement

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [soft-vc](https://github.com/bshall/soft-vc)

- [ContentVec](https://github.com/auspicious3000/contentvec)

- [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

- [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)

- [whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc)

- [Whisper](https://github.com/openai/whisper)

- [SpeechBrain](https://github.com/speechbrain/speechbrain)

- [FAISS](https://github.com/facebookresearch/faiss)

- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)