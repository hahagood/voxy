"""文本输出模块 - 剪贴板/直接输入/标准输出"""

import os
import shutil
import subprocess
import sys


def _is_wayland() -> bool:
    """检测当前是否运行在 Wayland 下。"""
    return os.environ.get("WAYLAND_DISPLAY") is not None


def _copy_to_clipboard(text: str) -> None:
    """复制文本到剪贴板，自动检测 Wayland/X11。"""
    if _is_wayland():
        if not shutil.which("wl-copy"):
            raise RuntimeError("Wayland 剪贴板需要安装 wl-clipboard: pacman -S wl-clipboard")
        subprocess.run(["wl-copy", text], check=True)
    else:
        if not shutil.which("xclip"):
            raise RuntimeError("X11 剪贴板需要安装 xclip: pacman -S xclip")
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            check=True,
        )


def _type_text(text: str) -> None:
    """模拟键盘输入文本到当前焦点窗口。"""
    if _is_wayland():
        if not shutil.which("wtype"):
            raise RuntimeError("Wayland 文字输入需要安装 wtype: pacman -S wtype")
        subprocess.run(["wtype", "--", text], check=True)
    else:
        if not shutil.which("xdotool"):
            raise RuntimeError("X11 文字输入需要安装 xdotool: pacman -S xdotool")
        subprocess.run(["xdotool", "type", "--", text], check=True)


def output_text(text: str, mode: str) -> None:
    """根据模式输出文本。

    Args:
        text: 要输出的文本
        mode: 输出模式 - clipboard / stdout / type
    """
    if mode == "stdout":
        print(text)
    elif mode == "clipboard":
        _copy_to_clipboard(text + "\n")
        print("  已复制到剪贴板。", file=sys.stderr)
    elif mode == "type":
        _type_text(text + "\n")
        print("  已输入到焦点窗口。", file=sys.stderr)
    else:
        raise ValueError(f"未知的输出模式: {mode}，支持: clipboard / stdout / type")
