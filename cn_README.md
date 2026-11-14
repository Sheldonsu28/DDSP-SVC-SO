Language: [English](./README.md) **简体中文**

# DDSP-SVC

DDSP-SVC 是一个开源歌声转换项目。本分支在上游 DDSP / Rectified-Flow 基础上，提供 SoftVcStylizer 前端、完整音频判别器和 NSF-HifiGAN 的训练与推理流程。

与上游 DDSP 相比，本仓库改善了合成语音的发音表现，并提高了目标说话人音色的还原度，代价是需要更高的训练成本。

本次迁移主要支持以下四个入口：

| 入口 | 用途 |
| --- | --- |
| [`preprocess.py`](preprocess.py) | 提取 ContentVec、Whisper PPG、HuBERT Soft、说话人嵌入、音高、音量和梅尔频谱特征。 |
| [`train_frontend_vc_gan.py`](train_frontend_vc_gan.py) | 冻结基模型的 DDSP 参数，训练 SoftVcStylizer 前端与完整音频判别器。 |
| [`train_index.py`](train_index.py) | 分别建立 ContentVec、HuBERT 和 Whisper 的 FAISS 特征索引。 |
| [`main_reflow.py`](main_reflow.py) | 使用基模型、可选的 SoftVcStylizer 前端及特征检索进行音频转换。 |

ContentVec 使用目标代码库中的 Hugging Face 实现和 `pytorch_model.bin` 权重格式。BigVGAN、旧版 stylizer、Fairseq ContentVec、小型判别器及实验性残差模型不在本分支的迁移范围内。

发布代码不附带个人训练权重、说话人嵌入、数据集、预处理缓存或检索索引。这些资源由使用者自行准备或训练生成。中文前端操作说明见 [cn_FRONTEND.md](cn_FRONTEND.md)，英文版见 [FRONTEND.md](FRONTEND.md)，迁移与验证记录见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。

免责声明：请确保仅使用**合法获得的授权数据**训练 DDSP-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。本库作者不对因使用这些模型检查点和音频而造成的任何侵权、诈骗等违法行为负责。

## 1. 安装依赖

本分支已验证的环境为 **Windows + Python 3.11 + PyTorch 2.11.0 / Torchaudio 2.11.0 的 CUDA 12.6 版本**。在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

安装命令对应 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。可以用下面的命令检查环境：

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); print(torch.cuda.is_available())"
```

CUDA 环境中的两个版本号应为 `2.11.0+cu126`，最后一项应为 `True`。

`requirements.txt` 只列出当前四个入口所需的直接依赖。其中 `transformers` 用于新的 ContentVec 实现；SpeechBrain 用于 ECAPA 说话人嵌入；FAISS 与 scikit-learn 用于特征索引。Whisper PPG 使用仓库内的音频编码器，不需要额外安装 Whisper 文本解码工具。

默认音高提取器 RMVPE 不需要其他音高后端。只有在选择 Parselmouth、WORLD、CREPE 或 FCPE 时，才需要安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pitch.txt
```

GUI、实时音频设备工具等历史入口的依赖不包含在当前主依赖清单中。

## 2. 准备预训练资源

SoftVcStylizer 前端使用三路内容特征，因此需要同时准备 ContentVec、Whisper PPG 和 HuBERT Soft。

| 资源 | 默认位置或配置 | 说明 |
| --- | --- | --- |
| [ContentVec](https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin?download=true) | `pretrain/contentvec/pytorch_model.bin` | 使用上游转换后的 Hugging Face 权重，不使用 Fairseq 的旧 `.pt` 格式。 |
| Whisper large-v2 | `pretrain/ppg-large/large-v2.pt` | 标准权重文件应包含 `dims` 和 `model_state_dict`。PPG 提取保留音频编码器的前四分之三层，不使用文本解码器。 |
| [HuBERT Soft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) | `pretrain/hubert/hubert-soft.pt` | 下载后按此路径放置，或通过 `frontend.hubert_ckpt` 指定实际文件名。 |
| [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) | `pretrain/rmvpe/model.pt` | 默认音高提取器。 |
| [NSF-HifiGAN](https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip) | `vocoder.ckpt`，默认 `pretrain/nsf_hifigan/model` | 权重旁必须有匹配的 `config.json`。声码器采样率和 hop size 必须与数据配置一致。 |
| ECAPA 说话人编码器 | `speechbrain/spkrec-ecapa-voxceleb` | 启用前端预处理时由 SpeechBrain 获取，默认缓存到 `cache/speaker_encoder`；也可将 `frontend.speaker_encoder` 设置为本地模型目录。 |

声码器也可以通过 [SingingVocoders](https://github.com/openvpi/SingingVocoders) 自行训练或微调。上述模型文件、缓存和训练输出均不应加入发布代码。

## 3. 配置与数据预处理

以 [`configs/frontend-vc-gan.yaml`](configs/frontend-vc-gan.yaml) 为前端流程的完整配置起点。根据自己的数据修改 `data.train_path`、`data.valid_path`、`vocoder.ckpt` 和实验目录。

`configs/reflow.yaml` 保留为上游基线配置。不启用前端时，预处理和推理仍可使用该配置。

### 3.1. 数据集目录

单说话人数据集：

```text
data/
├─ train/
│  └─ audio/
│     ├─ example.wav
│     └─ ...
└─ val/
   └─ audio/
      ├─ validation.wav
      └─ ...
```

多说话人数据集需要设置 `model.n_spk`，并将音频放到 `audio/1/`、`audio/2/` 等目录。说话人目录名必须是 `1` 到 `n_spk` 之间的正整数，训练集与验证集应采用一致的编号。

训练切片需足够覆盖 `data.duration` 指定的训练长度。音频会按 `data.sampling_rate` 加载；提前统一采样率可以减少预处理中的重采样开销。`train.cache_all_data: false` 可避免把全部特征预先缓存到内存或显存中。

### 3.2. 前端资源配置

以下字段已经包含在完整配置中，可按需修改：

```yaml
frontend:
  enabled: true
  base_checkpoint: null          # 基模型权重或实验目录；默认使用 env.expdir
  checkpoint: null               # 前端推理必需；训练时可用于指定恢复权重
  speaker_embedding: null        # 可选推理覆盖；默认使用训练集对应说话人的均值
  index_dir: exp/feature_index   # 三路索引的输出/读取目录
  contentvec_index: null         # 可单独指定索引文件，覆盖 index_dir 中的默认路径
  hubert_index: null
  whisper_index: null
  whisper_ckpt: pretrain/ppg-large/large-v2.pt
  hubert_ckpt: pretrain/hubert/hubert-soft.pt
  speaker_encoder: speechbrain/spkrec-ecapa-voxceleb
  speaker_encoder_cache: cache/speaker_encoder
  model: {}                     # SoftVcStylizer 构造参数；与前端检查点一起保存
```

显式提供的命令行资源路径优先于 YAML。推理时可用 `--frontend_checkpoint`、`--speaker_embedding`、`--index_dir`，或者单独的 `--contentvec_index`、`--hubert_index`、`--whisper_index` 指定资源。

前端预处理按说话人编号，使用成功处理的切片计算 ECAPA 嵌入算术均值，并以 FP32 保存到数据集目录下的 `speaker_means/<speaker_id>.npy`。训练时每个说话人的均值文件只加载一次，验证集复用训练集均值；缓存和非缓存模式均适用，不再使用全局均值文件或相似度阈值。启用前端推理时，`main_reflow.py` 自动从 `data.train_path/speaker_means/<spk_id>.npy` 加载均值，使用 `-id` / `--spk_id`（默认 1）选择说话人，在处理音频前只加载一次。`-c` 配置中的 `data.train_path` 优先于模型保存的训练路径。可选的 `speaker_embedding` 配置或 `--speaker_embedding` 参数仅用于推理，可指定均值文件或目录以覆盖默认路径；指定目录时同样按 `<spk_id>.npy` 选择文件。

### 3.3. 执行预处理

```powershell
.\.venv\Scripts\python.exe preprocess.py -c configs/frontend-vc-gan.yaml --store_dtype fp16 -j 1
```

- 前端模式默认使用 1 个工作进程；不启用前端的上游流程默认使用 2 个。`-j` / `--workers` 可以覆盖此设置。每个进程都会各自加载编码器。
- `--store_dtype fp16` 将内容特征和梅尔特征保存为 FP16；也可以选择默认的 `fp32`。
- F0、音量和说话人嵌入始终保存为 FP32。`cache_fp16: false` 时，加载器会把 FP16 特征恢复为 FP32，验证集也采用此处理。
- 训练集根据 `model.use_pitch_aug` 进行增强；验证集不进行音高增强。
- F0 提取失败的音频会移入对应数据集的 `skip/` 目录。

启用前端后，生成的主要目录包括：

| 目录 | 内容 |
| --- | --- |
| `units/`、`whisper_units/`、`hubert_units/` | 分别为 768、1280、256 维内容特征。 |
| `units_aug/`、`whisper_units_aug/`、`hubert_units_aug/` | 增强后的内容特征。 |
| `mel/`、`aug_mel/` | 常规梅尔频谱及其增强版本。 |
| `speaker/` | 每个切片的 192 维 ECAPA 嵌入。 |
| `speaker_means/` | 按说话人编号保存的 FP32 均值，例如 `1.npy`；训练每个文件只读取一次。 |
| `f0/`、`volume/`、`aug_vol/` | 音高、音量和增强音量。 |

数据集根目录还会生成 `pitch_aug_dict.npy`。本分支应重新进行预处理，并重新建立索引，不以旧版缓存和个人训练资源为兼容目标。

## 4. 训练 SoftVcStylizer 前端

前端训练需要一个与当前特征配置匹配的 DDSP / Rectified-Flow 基模型。基模型可通过现有的 `train_reflow.py` 训练；其采样率、`block_size`、编码器名称及输出维度必须与前端配置一致，其他特征提取参数也应保持一致。

`env.expdir` 指向基模型实验目录，`env.style_dir` 指向前端检查点和日志目录。完整配置默认分别为 `exp/reflow-test` 与 `exp/frontend`。

```powershell
.\.venv\Scripts\python.exe train_frontend_vc_gan.py -c configs/frontend-vc-gan.yaml --base_checkpoint exp/reflow-test/model_10000.pt
```

示例中的权重编号需替换为自己的实际文件。训练器只加载并冻结基模型的 DDSP 部分，保留从生成音频到前端的梯度；训练 SoftVcStylizer 前端与完整音频判别器。默认前端训练配置使用 BF16。

再次运行相同命令，会从 `env.style_dir` 中编号最大的前端检查点恢复。也可以指定某个检查点：

```powershell
.\.venv\Scripts\python.exe train_frontend_vc_gan.py -c configs/frontend-vc-gan.yaml --base_checkpoint exp/reflow-test/model_10000.pt --frontend_checkpoint exp/frontend/model_500.pt
```

`frontend.checkpoint` 也可指定恢复权重；新的检查点仍写入 `env.style_dir`。`train.save_opt: true` 时保存优化器状态，以便恢复。检查点保存频率由 `interval_val` 和 `interval_force_save` 控制。

## 5. 建立特征检索索引

完成前端预处理后运行：

```powershell
.\.venv\Scripts\python.exe train_index.py -c configs/frontend-vc-gan.yaml --output_dir exp/feature_index
```

程序从配置中的训练集和验证集读取特征，转换为连续的 FP32 数组后建立 FAISS 索引，因此也支持 FP16 预处理文件。输出文件为：

| 文件 | 特征流 |
| --- | --- |
| `feature_and_index.pkl` | ContentVec |
| `feature_and_index_hubert.pkl` | HuBERT Soft |
| `feature_and_index_whisper.pkl` | Whisper PPG |

`--output_dir` 也可写为 `--index_dir`；未提供时使用 `frontend.index_dir`。`--root_dir` 可覆盖配置中的数据路径，按该目录下的 `train/`、`val/` 查找特征。

单说话人默认使用索引字典键 `1`。多说话人需分别指定输入子目录和输出目录，例如：

```powershell
.\.venv\Scripts\python.exe train_index.py -c configs/frontend-vc-gan.yaml --speaker_id 1 --speaker_subdir 1 --output_dir exp/feature_index/1
```

其他说话人使用自己的编号与输出目录重复运行，推理时选择对应目录。这样可避免把多个说话人的特征混入同一个索引或覆盖已有输出。

## 6. 推理

下面的前端推理示例显式指定基模型、前端权重和索引目录，并从训练路径自动加载说话人均值：

```powershell
.\.venv\Scripts\python.exe main_reflow.py `
  -c configs/frontend-vc-gan.yaml `
  -m exp/reflow-test/model_10000.pt `
  -i input.wav -o output.wav `
  --frontend_checkpoint exp/frontend/model_500.pt `
  -id 1 `
  --index_dir exp/feature_index `
  -fr 1
```

这些资源路径也可以写入 `frontend` 配置。多说话人场景用 `-id` 指定说话人编号，均值从训练路径自动选择；特征检索仍需选择对应说话人的索引目录。

`-fr 0` 关闭特征检索。沿用源代码的行为，正值表示启用分布映射，数值大小不是插值权重；映射仍使用原有的标准差公式。

常用参数包括 `-k`（移调半音数）、`-f`（共振峰偏移）、`-pe`（音高提取器）、`-step`（整流流采样步数）、`-method`（`euler` / `rk4`）和 `-ts`（采样起始时间）。`-v` 的声区调整需要所选声码器支持。完整参数说明可通过以下命令查看：

```powershell
.\.venv\Scripts\python.exe main_reflow.py --help
```

普通上游推理可以省略前端配置，或显式关闭前端：

```powershell
.\.venv\Scripts\python.exe main_reflow.py -m exp/reflow-test/model_10000.pt -i input.wav -o output.wav --no_frontend
```

## 7. 训练日志与验证

TensorBoard 日志默认写入前端实验目录下的 `logs/`：

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir exp/frontend/logs
```

运行回归检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

流程已在独立 Python 3.11 环境通过 29 项回归检查，并验证了实际编码器、单进程/多进程预处理、三路索引、检索开关推理以及前端训练与恢复。测试使用临时数据和新初始化的检查点，用于验证计算和流程；实际音色质量需要使用自己的数据训练后评估。

本次迁移包含两个明确修正：

1. 音频裁切起点和长度使用配置中的输出 `block_size`，与已经对齐的缓存特征保持一致；默认配置按 512 个采样点一帧计算。
2. 说话人编码器接收单独重采样至 16 kHz 的输入，并使用所选设备；其他特征继续使用原始训练采样率的音频。

同时保留上游的 STFT / ONNX 填充修正、预处理多进程和 FP16 存储支持。检索映射的数值行为保持不变。

## 8. 感谢

- [ddsp](https://github.com/magenta/ddsp)
- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
- [soft-vc](https://github.com/bshall/soft-vc)
- [ContentVec](https://github.com/auspicious3000/contentvec)
- [DiffSinger / OpenVPI](https://github.com/openvpi/DiffSinger)
- [Diff-SVC](https://github.com/prophesier/diff-svc)
- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)
- [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
- [Whisper](https://github.com/openai/whisper)
- [whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
