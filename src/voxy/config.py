"""TOML 配置管理"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_PATH = Path.home() / ".config" / "voxy" / "config.toml"

DEFAULTS: dict[str, Any] = {
    "audio": {
        "device": "default",
        "sample_rate": 16000,
        "silence_threshold": 0.15,
        "silence_duration": 2.0,
    },
    "stt": {
        "backend": "sensevoice",
        "language": "auto",
        "whisper": {
            "model": "small",
            "device": "cuda",
            "compute_type": "int8",
        },
        "sensevoice": {
            "model": "iic/SenseVoiceSmall",
            "device": "cuda:0",
        },
        "cloud": {
            "api_base": "https://api.openai.com/v1",
            "api_key": "",
            "model": "whisper-1",
        },
    },
    "llm": {
        "enabled": False,
        "provider": "ollama/qwen2.5:3b-instruct",
        "api_base": "http://localhost:11434",
        "api_key": "",
        "proxy": "",
        "custom_terms": {},
    },
    "daemon": {
        "enabled": True,
        "idle_timeout": 10,
    },
    "output": {
        "mode": "clipboard",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AudioConfig:
    device: str = "default"
    sample_rate: int = 16000
    silence_threshold: float = 0.15
    silence_duration: float = 2.0


@dataclass
class WhisperConfig:
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "int8"


@dataclass
class SenseVoiceConfig:
    model: str = "iic/SenseVoiceSmall"
    device: str = "cuda:0"


@dataclass
class CloudSTTConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "whisper-1"


@dataclass
class STTConfig:
    backend: str = "sensevoice"
    language: str = "auto"
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    cloud: CloudSTTConfig = field(default_factory=CloudSTTConfig)


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "ollama/qwen2.5:3b-instruct"
    api_base: str = "http://localhost:11434"
    api_key: str = ""
    proxy: str = ""
    custom_terms: dict[str, str] = field(default_factory=dict)


@dataclass
class DaemonConfig:
    enabled: bool = True
    idle_timeout: int = 10


@dataclass
class OutputConfig:
    mode: str = "clipboard"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _build_config(data: dict) -> Config:
    """Build a Config object from a flat dict."""
    audio_d = data.get("audio", {})
    stt_d = data.get("stt", {})
    llm_d = data.get("llm", {})
    daemon_d = data.get("daemon", {})
    output_d = data.get("output", {})

    return Config(
        audio=AudioConfig(**{k: v for k, v in audio_d.items() if not isinstance(v, dict)}),
        stt=STTConfig(
            backend=stt_d.get("backend", "sensevoice"),
            language=stt_d.get("language", "auto"),
            whisper=WhisperConfig(**stt_d.get("whisper", {})),
            sensevoice=SenseVoiceConfig(**stt_d.get("sensevoice", {})),
            cloud=CloudSTTConfig(**stt_d.get("cloud", {})),
        ),
        llm=LLMConfig(
            **{k: v for k, v in llm_d.items() if k != "custom_terms"},
            custom_terms=llm_d.get("custom_terms", {}),
        ),
        daemon=DaemonConfig(**daemon_d),
        output=OutputConfig(**output_d),
    )


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, merging with defaults."""
    config_path = path or CONFIG_PATH

    if config_path.exists():
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        merged = _deep_merge(DEFAULTS, user_config)
    else:
        merged = DEFAULTS.copy()

    return _build_config(merged)
