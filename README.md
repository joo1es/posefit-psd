# PoseFit-PSD 👤👔

> **智能姿态与骨骼对齐的头身拼贴工具，一键生成标准 5 图层 PSD 与合成预览图，专为后续 AI 局部重绘融合（Inpaint / ControlNet）打造。**

---

## 🌟 核心特性

- **零畸变刚性与躯干自适应对齐（Rigid & Dual-Axis Width Fitting）**：
  - 彻底摒弃粗暴的全局网格拉扯，交叉手、交叉腿、手指与服装褶皱 100% 自然保持原貌。
  - **肩胯双轴优化**：上锚定肩宽，下平滑过渡拟合图A腰胯轮廓，胖瘦体型自适应协调。
- **全自动模式适配（半身 / 全身无缝支持）**：
  - **全身照**：自动检测脚踝接触线，锁定脚底接地线（Ground Line Alignment），高跟鞋/鞋底稳稳踩在地面上，绝无悬空浮空感。
  - **半身/中景照**：自动对齐腰胯高度与皮带位置；胸像特写自动延伸躯干至画面底部。
- **底图人像遮罩擦除（Inpaint Clean Background）**：
  - 自动将图A原本露在后面的手脚、旧衣服在底图上通过算法智能抹除，彻底根治“露底图四条腿/多余肢体”问题。
- **专为 AI 重绘融合设计的 5 图层 PSD 架构**：
  - 直接输出分层 `.psd` 文件，在 Photoshop 中层级从顶至底依次为：
    1. `Photo A Head`：图A原生头部与发丝（显示）
    2. `Photo B Body`：对齐贴地/贴腰的图B身体层（显示）
    3. `Photo B Body 不rmbg`：带原背景的图B身体（隐藏备份，保留原始光影细节）
    4. `Clean Background`：已抹除原躯干与四肢的纯净背景底图（显示）
    5. `原图`：人像A原始完整照片（隐藏备份）

---

## 🛠️ 快速上手

### 1. 环境准备

推荐使用 [uv](https://github.com/astral-sh/uv) 极速配置环境：

```bash
# 安装 uv 并创建虚拟环境
pip install uv
uv venv .venv
.venv\Scripts\activate   # Windows

# 安装核心依赖
uv pip install mediapipe pytoshop scipy pillow opencv-python rembg onnxruntime psd-tools packbits
```

### 2. 下载 MediaPipe 姿态模型

确保将 `pose_landmarker.task` 放置在 `models/` 目录下：
```bash
mkdir -p models
curl -L -o models/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

### 3. 一键运行

```bash
python run_swap.py --target "path/to/photo_a.jpg" --donor "path/to/photo_b.jpg" --output "my_result"
```

- `--target`: 人像照片A（保留头部、发丝与环境底图）
- `--donor`: 人像照片B（提取身体、服装与姿态）
- `--output`: 输出文件名前缀，将自动生成 `my_result.psd` 与 `my_result.png`。

---

## 🎨 与 AI 编辑模型工作流配合

1. **第一步（本项目负责）**：快速完成骨骼与几何级粗对齐，输出标准 5 图层 PSD。
2. **第二步（AI 模型负责）**：在 WebUI / ComfyUI / Photoshop Generative Fill 中：
   - 使用蒙版选中**颈部接缝、领口及边缘过渡区**；
   - 配合 ControlNet (OpenPose / Inpaint) 或 SD / Flux Fill 进行轻微重绘（Denoise 0.4~0.6）；
   - AI 将自动利用保留的头部光影与服装纹理，实现极其自然、工业级的终极光影与接缝融合！

---

## 📄 License

MIT License
