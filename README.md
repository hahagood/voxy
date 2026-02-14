# Voxy

Linux 语音听写工具 — Typeless 开源平替。

按下快捷键，对着麦克风说话，文字自动输入到当前焦点窗口。

## 特性

- **SenseVoice STT** — 非自回归架构，推理极快（RTF ~0.05），中文效果优秀
- **多 STT 后端** — SenseVoice / faster-whisper / OpenAI Whisper API，可插拔切换
- **AI 润色（可选）** — 通过 litellm 调用任意 LLM（Ollama / OpenAI 等）
- **自动静音检测** — 录音前自动测量环境噪音，动态设定阈值，说完自动停止
- **智能输入** — 自动识别窗口类型，选择最佳输入方式（粘贴 / 直接输入）
- **Hyprland 集成** — 全局快捷键 `Super+R` 弹出浮动小窗录音

## 系统要求

- Arch Linux / Wayland (Hyprland)
- Python 3.12+
- CUDA GPU（推荐，也支持 CPU）

## 安装

### 系统依赖 (Arch Linux)

```bash
# 必需
sudo pacman -S portaudio       # sounddevice 音频采集
sudo pacman -S wtype           # Wayland 文字输入
sudo pacman -S wl-clipboard    # Wayland 剪贴板 (wl-copy)

# XWayland 应用支持（Emacs X11 等）
sudo pacman -S xdotool         # X11 文字输入
```

### Python 依赖

```bash
# 克隆项目
git clone https://github.com/hahagood/voxy.git
cd voxy

# 创建虚拟环境（需要 Python 3.12，系统若为 3.14 需指定版本）
uv venv --python 3.12

# 安装核心依赖
uv sync

# 安装 SenseVoice STT（推荐）
uv pip install funasr torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 faster-whisper STT（可选）
uv sync --extra whisper
```

## 使用

### 命令行

```bash
uv run voxy record --raw -o stdout    # 录音 → 转写 → 终端输出
uv run voxy record --raw              # 录音 → 转写 → 剪贴板
uv run voxy record --raw -o type      # 录音 → 转写 → 输入到焦点窗口
uv run voxy devices                   # 列出音频设备
uv run voxy config                    # 显示当前配置
```

### Hyprland 全局快捷键

将 `voxy-record` 脚本放到 `~/.local/bin/`，在 `hyprland.conf` 中添加：

```conf
bind = $mod, R, exec, voxy-record

windowrule {
    name = voxy-float
    match:class = ^(voxy-float)$
    float = yes
    size = 320 80
    move = 2230 45
}
```

按 `Super+R`：弹出浮动小窗 → 说话 → 静音自动停止 → 转写结果自动输入到之前的焦点窗口。

`voxy-record` 脚本会自动检测焦点窗口类型，选择最佳输入方式：

| 窗口类型 | 输入方式 |
|----------|----------|
| 浏览器、GUI 应用 | Ctrl+V 粘贴 |
| 终端（foot/kitty 等） | Ctrl+Shift+V 粘贴 |
| Emacs (Wayland) | wtype 直接输入 |
| XWayland 应用 | xdotool 逐字输入 |

## 配置

复制 `config.example.toml` 到 `~/.config/voxy/config.toml`：

```bash
mkdir -p ~/.config/voxy
cp config.example.toml ~/.config/voxy/config.toml
```

主要配置项：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `stt.backend` | `sensevoice` | STT 后端：sensevoice / whisper / cloud |
| `stt.language` | `auto` | 识别语言：auto / zh / en / ja ... |
| `llm.enabled` | `false` | 是否启用 AI 文本润色 |
| `llm.provider` | `ollama/qwen3:4b` | litellm 模型标识 |
| `output.mode` | `clipboard` | 输出方式：clipboard / stdout / type |

## 项目结构

```
src/voxy/
├── cli.py           # CLI 入口 (click)
├── config.py        # TOML 配置管理
├── audio.py         # 麦克风录音 + 动态静音检测
├── stt/
│   ├── __init__.py      # STT 基类 + 工厂函数
│   ├── local_sense.py   # SenseVoice (funasr)
│   ├── local_whisper.py # faster-whisper
│   └── cloud.py         # OpenAI Whisper API
├── processor.py     # AI 文本润色 (litellm)
├── prompts.py       # LLM 提示词模板
└── output.py        # 文本输出 (wtype/剪贴板/stdout)
```

## License

MIT
