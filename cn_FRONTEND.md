Language: [English](FRONTEND.md) **简体中文**

本分支支持 `preprocess.py`、`main_reflow.py`、`train_frontend_vc_gan.py` 和 `train_index.py`，使用 SoftVcStylizer 前端、完整音频判别器、NSF-HifiGAN，以及上游的 Hugging Face ContentVec 编码器。BigVGAN、旧版 stylizer、Fairseq 和实验性残差模型不在支持范围内。发布代码不附带个人检查点、说话人嵌入、数据集或检索索引。

在 Windows 上使用 Python 3.11，先安装匹配的 CUDA 版 PyTorch 和 Torchaudio，再安装项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CUDA 安装命令对应 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。Torch 和 Torchaudio 均应显示 `2.11.0+cu126`，`torch.cuda.is_available()` 应返回 `True`。`requirements-pitch.txt` 为 Parselmouth、WORLD、CREPE 和 FCPE 的可选依赖；默认 RMVPE 流程不需要这些包。GUI 和实时音频设备工具不在当前依赖清单的覆盖范围内。

请自行在本地准备以下公共预训练资源：

| 资源 | 默认路径或配置 | 说明 |
| --- | --- | --- |
| Hugging Face ContentVec | `pretrain/contentvec/pytorch_model.bin` | 使用上游转换后的检查点，不使用 Fairseq 的旧 `.pt` 格式。 |
| [Whisper large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) | `pretrain/ppg-large/large-v2.pt` | 标准检查点需包含 `dims` 和 `model_state_dict`；PPG 加载器保留音频编码器的前四分之三层，不需要文本解码。 |
| HuBERT Soft | `pretrain/hubert/hubert-soft.pt` | 使用与仓库内实现匹配的公共 HuBERT Soft 权重。 |
| RMVPE | `pretrain/rmvpe/model.pt` | 默认音高提取器。 |
| NSF-HifiGAN | YAML 中的 `vocoder.ckpt` | 匹配的 `config.json` 必须放在权重旁边。采样率和 hop size 必须与数据配置一致。 |
| ECAPA 说话人编码器 | `speechbrain/spkrec-ecapa-voxceleb` | 显式运行前端预处理时由 SpeechBrain 下载，缓存目录由 `frontend.speaker_encoder_cache` 指定；`frontend.speaker_encoder` 也可指向本地模型目录。 |

从 [`configs/frontend-vc-gan.yaml`](configs/frontend-vc-gan.yaml) 开始配置，设置训练/验证音频路径、声码器检查点和基模型实验目录。基模型的采样率、`block_size`、编码器名称及特征维度必须匹配。现有的上游 `train_reflow.py` 可用于训练基模型；前端训练器仅提取并冻结其中的 DDSP 部分。

前端资源配置如下：

```yaml
frontend:
  enabled: true
  base_checkpoint: null          # 默认使用 env.expdir；也可指定检查点文件或目录
  checkpoint: null               # 前端推理必需；训练时可指定恢复检查点
  speaker_embedding: null        # 可选推理覆盖；默认使用训练集对应说话人的均值
  index_dir: exp/feature_index   # 三路索引的输出/读取目录
  contentvec_index: null         # 单独指定文件时，覆盖 index_dir 下的默认路径
  hubert_index: null
  whisper_index: null
  whisper_ckpt: pretrain/ppg-large/large-v2.pt
  hubert_ckpt: pretrain/hubert/hubert-soft.pt
  model: {}                     # SoftVcStylizer 构造参数；随检查点配置保存
```

显式提供的命令行资源路径优先于 YAML。前端预处理按说话人编号，仅使用成功处理的训练切片计算 ECAPA 嵌入算术均值，并将相同的 FP32 均值分别保存到 `data.train_path/speaker_means/<speaker_id>.npy` 和 `data.valid_path/speaker_means/<speaker_id>.npy`。验证切片不参与均值计算。训练时每个说话人的均值文件只加载一次，验证集复用训练集均值；缓存和非缓存模式均适用。不再使用全局均值文件或相似度阈值。启用前端推理时，`main_reflow.py` 自动从 `data.train_path/speaker_means/<spk_id>.npy` 加载均值，使用 `-id` / `--spk_id`（默认 1）选择说话人，在处理音频前只加载一次。`-c` 配置中的 `data.train_path` 优先于模型保存的训练路径。可选的 `speaker_embedding` 配置或 `--speaker_embedding` 参数仅用于推理，可指定均值文件或目录以覆盖默认路径；指定目录时同样按 `<spk_id>.npy` 选择文件。

训练器默认恢复 `env.style_dir` 中编号最大的检查点。`frontend.checkpoint` 或 `--frontend_checkpoint` 可选择某个 GAN 检查点；新的输出仍写入 `env.style_dir`。

在仓库根目录运行以下流程。示例中的检查点路径和编号需替换为自己的实际文件：

```powershell
# 在 data/train/audio 和 data/val/audio 中放入音频，然后提取全部前端特征。
.\.venv\Scripts\python.exe preprocess.py -c configs/frontend-vc-gan.yaml --store_dtype fp16 -j 1

# 基于自己的 DDSP 基模型训练前端，恢复目录由 env.style_dir 指定。
.\.venv\Scripts\python.exe train_frontend_vc_gan.py -c configs/frontend-vc-gan.yaml --base_checkpoint exp/base/model_10000.pt

# 从配置中的训练集/验证集建立 ContentVec、HuBERT 和 Whisper 索引。
.\.venv\Scripts\python.exe train_index.py -c configs/frontend-vc-gan.yaml --output_dir exp/feature_index

# 显式指定基模型和前端检查点，从训练路径自动加载说话人均值，启用检索进行转换。
.\.venv\Scripts\python.exe main_reflow.py -m exp/base/model_10000.pt -i input.wav -o output.wav -c configs/frontend-vc-gan.yaml --frontend_checkpoint exp/frontend/model_500.pt -id 1 --index_dir exp/feature_index -fr 1
```

`-fr 0` 关闭检索。沿用源代码行为，正值表示启用分布映射，数值大小不是插值权重。检索保留原有的标准差公式，包括特征通道为常量时的数值行为。也可通过 `--contentvec_index`、`--hubert_index` 和 `--whisper_index` 分别指定索引文件。

索引保留原有的说话人字典格式与文件名：ContentVec 对应 `feature_and_index.pkl`，HuBERT 对应 `feature_and_index_hubert.pkl`，Whisper 对应 `feature_and_index_whisper.pkl`。`train_index.py` 的 `--speaker_id` 控制字典中的说话人键，`--speaker_subdir` 控制输入特征的说话人子目录。多说话人数据应分别指定子目录和输出目录，具体示例见 [中文 README](cn_README.md)。

如果只进行普通上游推理，可省略前端设置，或传入 `--no_frontend`。[`configs/reflow.yaml`](configs/reflow.yaml) 保留为上游基线配置。预处理在上游模式下默认使用两个工作进程，在前端模式下默认使用一个，因为每个进程都会各自加载全部三路编码器；`--workers` 可覆盖该设置。

FP16 存储应用于内容特征和梅尔特征，F0、音量和说话人嵌入始终保存为 FP32。`cache_fp16` 为 `false` 时，数据加载器会把特征恢复为 FP32，验证集也采用该处理。

本分支包含两个明确修正：音频裁切使用配置中的输出 `block_size`；ECAPA 输入在所选设备上单独重采样至 16 kHz。原始训练采样率的音频保持不变。请为本分支重新预处理数据集；旧缓存、说话人嵌入和索引不属于兼容目标。

运行回归检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用合成样本与临时检查点。在资源齐备时，公共预训练模型的冒烟检查会运行真实的编码器和声码器计算；这些检查验证计算流程，不代表训练后的音色质量。