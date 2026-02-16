"""Daemon 客户端 - 通过 Unix socket 与守护进程通信"""

import json
import socket
import struct

import numpy as np

from voxy.daemon import get_socket_path


def _send_command(cmd: str) -> dict:
    """发送特殊命令到 daemon，返回响应。"""
    sock_path = get_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(sock_path)
        header = json.dumps({"cmd": cmd}).encode("utf-8")
        sock.sendall(struct.pack(">I", len(header)) + header)
        sock.shutdown(socket.SHUT_WR)

        resp_len_bytes = sock.recv(4)
        if len(resp_len_bytes) < 4:
            raise ConnectionError("响应不完整")
        resp_len = struct.unpack(">I", resp_len_bytes)[0]

        resp_bytes = bytearray()
        while len(resp_bytes) < resp_len:
            chunk = sock.recv(resp_len - len(resp_bytes))
            if not chunk:
                break
            resp_bytes.extend(chunk)

        return json.loads(resp_bytes.decode("utf-8"))
    finally:
        sock.close()


def daemon_ping() -> bool:
    """检测 daemon 是否可用。"""
    try:
        resp = _send_command("ping")
        return resp.get("ok", False)
    except Exception:
        return False


def daemon_status() -> dict | None:
    """获取 daemon 状态信息，不可用时返回 None。"""
    try:
        return _send_command("status")
    except Exception:
        return None


def daemon_shutdown() -> bool:
    """通知 daemon 关闭。"""
    try:
        resp = _send_command("shutdown")
        return resp.get("ok", False)
    except Exception:
        return False


def transcribe_via_daemon(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """通过 daemon 进行语音转写。

    Raises:
        Exception: daemon 不可用或转写失败时抛出
    """
    sock_path = get_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(60.0)  # 转写可能耗时（首次加载模型）
    try:
        sock.connect(sock_path)

        # 发送 header
        header = json.dumps({"sample_rate": sample_rate}).encode("utf-8")
        sock.sendall(struct.pack(">I", len(header)) + header)

        # 发送音频数据
        sock.sendall(audio.astype(np.float32).tobytes())
        sock.shutdown(socket.SHUT_WR)

        # 接收响应
        resp_len_bytes = sock.recv(4)
        if len(resp_len_bytes) < 4:
            raise ConnectionError("响应不完整")
        resp_len = struct.unpack(">I", resp_len_bytes)[0]

        resp_bytes = bytearray()
        while len(resp_bytes) < resp_len:
            chunk = sock.recv(resp_len - len(resp_bytes))
            if not chunk:
                break
            resp_bytes.extend(chunk)

        resp = json.loads(resp_bytes.decode("utf-8"))

        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "未知错误"))

        return resp.get("text", "")
    finally:
        sock.close()
