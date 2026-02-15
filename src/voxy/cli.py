"""CLI 入口 - click 命令行界面"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from voxy.config import load_config


HISTORY_PATH = Path.home() / ".local" / "share" / "voxy" / "history.json"


def _append_history(raw: str, polished: str) -> None:
    """追加一条转写/润色对照记录到 history.json"""
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if HISTORY_PATH.exists():
            records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        records.append(
            {
                "raw": raw,
                "polished": polished,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        HISTORY_PATH.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        click.echo(f"保存历史记录失败: {e}", err=True)


@click.group()
@click.pass_context
def main(ctx):
    """Voxy - Linux 语音听写工具"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()


@main.command()
@click.option("--raw", is_flag=True, help="跳过 AI 润色，直接输出原始转写")
@click.option(
    "-o",
    "--output",
    type=click.Choice(["clipboard", "stdout", "type"]),
    default=None,
    help="输出方式 (默认使用配置文件设置)",
)
@click.pass_context
def record(ctx, raw: bool, output: str | None):
    """录音 → 转写 → 润色 → 输出"""
    config = ctx.obj["config"]
    output_mode = output or config.output.mode

    # 1. 录音
    from voxy.audio import record as do_record

    try:
        audio_data = do_record(config.audio)
    except Exception as e:
        click.echo(f"录音失败: {e}", err=True)
        sys.exit(1)

    if audio_data.size == 0:
        click.echo("未录到音频。", err=True)
        sys.exit(1)

    # 2. 语音识别
    from voxy.stt import create_stt

    click.echo("  转写中...", err=True)
    try:
        engine = create_stt(config.stt)
        text = engine.transcribe(audio_data, sample_rate=config.audio.sample_rate)
    except Exception as e:
        click.echo(f"转写失败: {e}", err=True)
        sys.exit(1)

    if not text.strip():
        click.echo("未识别到文字。", err=True)
        sys.exit(1)

    click.echo(f"  原始转写: {text}", err=True)

    # 3. AI 润色 (可选)
    if not raw and config.llm.enabled:
        from voxy.processor import process_text

        raw_text = text
        click.echo("  AI 润色中...", err=True)
        try:
            text = process_text(text, config.llm)
        except Exception as e:
            click.echo(f"AI 润色失败 (使用原始文本): {e}", err=True)
        else:
            _append_history(raw_text, text)

    # 4. 输出
    from voxy.output import output_text

    try:
        output_text(text, output_mode)
    except Exception as e:
        click.echo(f"输出失败: {e}", err=True)
        sys.exit(1)


@main.command()
def devices():
    """列出音频输入设备"""
    from voxy.audio import list_devices

    click.echo(list_devices())


@main.command("config")
@click.pass_context
def show_config(ctx):
    """显示当前配置"""
    from voxy.config import CONFIG_PATH

    config = ctx.obj["config"]

    click.echo(f"配置文件: {CONFIG_PATH}")
    click.echo(f"  存在: {'是' if CONFIG_PATH.exists() else '否 (使用默认配置)'}")
    click.echo()
    click.echo(f"[audio]")
    click.echo(f"  device = {config.audio.device}")
    click.echo(f"  sample_rate = {config.audio.sample_rate}")
    click.echo(f"  silence_threshold = {config.audio.silence_threshold}")
    click.echo(f"  silence_duration = {config.audio.silence_duration}")
    click.echo()
    click.echo(f"[stt]")
    click.echo(f"  backend = {config.stt.backend}")
    click.echo(f"  language = {config.stt.language}")

    if config.stt.backend == "whisper":
        click.echo(f"  [stt.whisper]")
        click.echo(f"    model = {config.stt.whisper.model}")
        click.echo(f"    device = {config.stt.whisper.device}")
        click.echo(f"    compute_type = {config.stt.whisper.compute_type}")
    elif config.stt.backend == "sensevoice":
        click.echo(f"  [stt.sensevoice]")
        click.echo(f"    model = {config.stt.sensevoice.model}")
        click.echo(f"    device = {config.stt.sensevoice.device}")
    elif config.stt.backend == "cloud":
        click.echo(f"  [stt.cloud]")
        click.echo(f"    api_base = {config.stt.cloud.api_base}")
        click.echo(f"    model = {config.stt.cloud.model}")
        click.echo(f"    api_key = {'***' if config.stt.cloud.api_key else '(未设置)'}")

    click.echo()
    click.echo(f"[llm]")
    click.echo(f"  enabled = {config.llm.enabled}")
    click.echo(f"  provider = {config.llm.provider}")
    click.echo(f"  api_base = {config.llm.api_base}")
    click.echo(f"  api_key = {'***' if config.llm.api_key else '(未设置)'}")
    click.echo()
    click.echo(f"[output]")
    click.echo(f"  mode = {config.output.mode}")
