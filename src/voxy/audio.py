"""麦克风录音模块 - sounddevice 流式采集 + 静音检测"""

import sys
import threading

import numpy as np
import sounddevice as sd

from voxy.config import AudioConfig


def list_devices() -> str:
    """列出所有音频输入设备。"""
    devices = sd.query_devices()
    lines = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            lines.append(f"  [{i}] {dev['name']} (输入通道: {dev['max_input_channels']}){marker}")
    if not lines:
        return "未找到音频输入设备"
    return "音频输入设备:\n" + "\n".join(lines) + "\n\n  * = 系统默认设备"


def _measure_noise(sample_rate: int, device, duration: float = 0.5) -> float:
    """测量环境噪音水平，返回平均音量。"""
    data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return float(np.abs(data).mean())


def record(config: AudioConfig) -> np.ndarray:
    """录音直到用户按 Enter 或连续静音超时。

    返回 16kHz 单声道 float32 numpy array。
    """
    sample_rate = config.sample_rate
    silence_duration = config.silence_duration
    device = None if config.device == "default" else config.device

    # 尝试将 device 字符串转换为整数索引
    if isinstance(device, str):
        try:
            device = int(device)
        except ValueError:
            pass

    # 自动测量环境噪音，阈值 = 噪音水平 * 1.5
    noise_level = _measure_noise(sample_rate, device)
    silence_threshold = noise_level * 1.5
    print(f"  环境噪音: {noise_level:.3f}, 静音阈值: {silence_threshold:.3f}", file=sys.stderr)

    chunks: list[np.ndarray] = []
    silent_samples = 0
    max_silent = int(silence_duration * sample_rate)
    has_speech = False  # 确保至少检测到一次说话才触发静音停止
    stop_event = threading.Event()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal silent_samples, has_speech
        if status:
            print(f"  音频警告: {status}", file=sys.stderr)

        chunk = indata[:, 0].copy()
        chunks.append(chunk)

        volume = np.abs(chunk).mean()
        if volume < silence_threshold:
            silent_samples += frames
            if has_speech and silent_samples >= max_silent:
                stop_event.set()
        else:
            silent_samples = 0
            has_speech = True

    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)

    print("  录音中... (按 Enter 停止，或静音自动停止)", file=sys.stderr)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        callback=audio_callback,
        blocksize=int(sample_rate * 0.1),  # 100ms blocks
    ):
        enter_thread.start()
        stop_event.wait()

    print("  录音结束。", file=sys.stderr)

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks)
