# Voxy

Linux 语音听写工具 — Typeless 开源平替。

按下快捷键，对着麦克风说话，文字自动输入到当前焦点窗口。

## 特性

- **SenseVoice STT** — 非自回归架构，推理极快（RTF ~0.05），中文效果优秀
- **多 STT 后端** — SenseVoice / faster-whisper / OpenAI Whisper API，可插拔切换
- **Daemon 模式** — STT 模型常驻 GPU 显存，systemd 开机自启，转写近乎瞬时
- **AI 润色（可选）** — 短文本走本地 Ollama（快），长文本自动切换云端大模型（质量好）
- **润色历史记录** — 自动保存原始转写与润色结果的对照记录，便于积累数据优化提示词
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
uv run voxy daemon start              # 启动 STT 守护进程（后台）
uv run voxy daemon status             # 查看守护进程状态
uv run voxy daemon stop               # 停止守护进程
uv run voxy devices                   # 列出音频设备
uv run voxy config                    # 显示当前配置
```

### Daemon 模式（推荐）

STT 模型常驻 GPU 显存，避免每次录音冷启动加载模型。通过 systemd 用户服务实现开机自启：

```bash
# 安装 systemd 服务（已含在 contrib/ 目录）
mkdir -p ~/.config/systemd/user
cp contrib/voxy-daemon.service ~/.config/systemd/user/

# 启用开机自启 + 立即启动
systemctl --user enable --now voxy-daemon.service

# 查看状态
systemctl --user status voxy-daemon.service
voxy daemon status
```

Daemon 首次收到转写请求时加载模型，之后常驻显存。空闲超过 `daemon.idle_timeout`（默认 10 分钟）自动卸载释放显存。`record` 命令会优先连接 daemon，不可用时自动回退直接模式。

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
| `llm.provider` | `ollama/qwen2.5:1.5b-instruct` | 短文本润色模型 |
| `llm.long_provider` | (空) | 长文本润色模型（如 `gemini/gemini-2.5-flash`） |
| `llm.long_threshold` | `200` | 超过 N 字切换到长文本模型 |
| `daemon.enabled` | `true` | 优先使用 daemon 转写 |
| `daemon.idle_timeout` | `10` | 空闲 N 分钟后自动卸载模型 |
| `output.mode` | `clipboard` | 输出方式：clipboard / stdout / type |

## 润色历史记录

当 AI 润色启用时，每次润色成功后会自动将原始转写和润色结果保存到：

```
~/.local/share/voxy/history.json
```

文件格式为 JSON 数组，每条记录包含：

```json
[
  {
    "raw": "原始转写文本",
    "polished": "AI 润色后文本",
    "timestamp": "2026-02-15T08:30:00+00:00"
  }
]
```

- `--raw` 模式或 `llm.enabled=false` 时不保存（无对照数据）
- 保存失败不影响主流程
- 可用 `jq` 查看：`jq . ~/.local/share/voxy/history.json`

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
├── daemon.py        # STT 守护进程 (Unix socket server)
├── daemon_client.py # 守护进程客户端
├── processor.py     # AI 文本润色 (Ollama / litellm)
├── prompts.py       # LLM 提示词模板
└── output.py        # 文本输出 (wtype/剪贴板/stdout)
```

## Roadmap

参考 [VoiceInk](https://github.com/Beingpax/VoiceInk) 等项目，计划改进：

- [x] **Daemon 模式** — 后台常驻，模型预加载，避免每次冷启动延迟
- [x] **自定义词汇表** — 配置常用术语和替换规则，提升中文专有名词识别准确率
- [ ] **多 Prompt Mode** — 支持 casual / formal / code 等多种润色风格，`--mode` 切换
- [ ] **Power Mode** — 根据当前焦点窗口 class 自动匹配润色规则
- [ ] **上下文感知** — 将剪贴板/选中文本作为上下文传给 LLM，提升润色质量
- [ ] **媒体播放控制** — 录音时自动暂停音乐（playerctl）

## License

MIT
