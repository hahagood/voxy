"""Voxy STT 守护进程 - 模型常驻内存，Unix socket 通信"""

import json
import os
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

from voxy.config import Config
from voxy.stt import STTEngine, create_stt


def get_socket_path() -> str:
    """返回 daemon socket 路径。"""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    sock_dir = os.path.join(runtime_dir, "voxy")
    os.makedirs(sock_dir, exist_ok=True)
    return os.path.join(sock_dir, "stt.sock")


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """从 socket 精确读取 n 字节。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接中断")
        buf.extend(chunk)
    return bytes(buf)


def _send_response(conn: socket.socket, data: dict) -> None:
    """发送 JSON 响应：[4 bytes 长度][JSON body]"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack(">I", len(body)) + body)


class DaemonServer:
    """STT 守护进程服务端。"""

    def __init__(self, config: Config):
        self._config = config
        self._engine: STTEngine | None = None
        self._engine_loaded = False
        self._last_active = time.monotonic()
        self._idle_timeout = config.daemon.idle_timeout * 60  # 分钟 → 秒
        self._running = False
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _ensure_engine(self) -> STTEngine:
        """确保 STT 引擎已加载。"""
        with self._lock:
            if self._engine is None:
                self._engine = create_stt(self._config.stt)
            if not self._engine_loaded:
                # 触发懒加载：用一小段静音做推理
                dummy = np.zeros(1600, dtype=np.float32)
                self._engine.transcribe(dummy, sample_rate=self._config.audio.sample_rate)
                self._engine_loaded = True
                print("  模型加载完成", file=sys.stderr, flush=True)
            self._last_active = time.monotonic()
            return self._engine

    def _unload_engine(self) -> None:
        """卸载模型释放显存。"""
        with self._lock:
            if self._engine is not None and self._engine_loaded:
                print("  空闲超时，卸载模型释放显存...", file=sys.stderr, flush=True)
                self._engine.unload()
                self._engine_loaded = False
                self._engine = None

    def _idle_watcher(self) -> None:
        """后台线程：检测空闲超时并卸载模型。"""
        while self._running:
            time.sleep(30)
            if not self._running:
                break
            with self._lock:
                idle_secs = time.monotonic() - self._last_active
                loaded = self._engine_loaded
            if loaded and idle_secs >= self._idle_timeout:
                self._unload_engine()

    def _handle_connection(self, conn: socket.socket) -> None:
        """处理单个客户端连接。"""
        try:
            # 读取 header 长度
            header_len_bytes = _recv_exact(conn, 4)
            header_len = struct.unpack(">I", header_len_bytes)[0]

            # 读取 JSON header
            header_bytes = _recv_exact(conn, header_len)
            header = json.loads(header_bytes.decode("utf-8"))

            # 检查是否是特殊命令
            cmd = header.get("cmd")
            if cmd == "ping":
                _send_response(conn, {"ok": True, "msg": "pong"})
                return
            elif cmd == "status":
                with self._lock:
                    loaded = self._engine_loaded
                    idle = time.monotonic() - self._last_active
                _send_response(conn, {
                    "ok": True,
                    "model_loaded": loaded,
                    "idle_seconds": round(idle, 1),
                    "backend": self._config.stt.backend,
                })
                return
            elif cmd == "shutdown":
                _send_response(conn, {"ok": True, "msg": "shutting down"})
                self._running = False
                return

            # 正常转写请求：读取剩余音频数据
            # 客户端发送完 header 后紧跟音频数据，然后关闭写端
            audio_chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                audio_chunks.append(chunk)

            if not audio_chunks:
                _send_response(conn, {"ok": False, "error": "未收到音频数据"})
                return

            audio_bytes = b"".join(audio_chunks)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            sample_rate = header.get("sample_rate", 16000)

            # 执行转写
            engine = self._ensure_engine()
            text = engine.transcribe(audio, sample_rate=sample_rate)

            _send_response(conn, {"ok": True, "text": text})

        except Exception as e:
            try:
                _send_response(conn, {"ok": False, "error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def _cleanup_stale_socket(self, sock_path: str) -> None:
        """检测并清理残留的 socket 文件。"""
        if not os.path.exists(sock_path):
            return
        # 尝试连接，如果失败说明是 stale socket
        test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            test_sock.connect(sock_path)
            test_sock.close()
            # 连接成功说明已有 daemon 在运行
            print(f"错误: 已有守护进程在运行 ({sock_path})", file=sys.stderr)
            sys.exit(1)
        except ConnectionRefusedError:
            # stale socket，删除
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        finally:
            test_sock.close()

    def run(self) -> None:
        """启动守护进程主循环。"""
        sock_path = get_socket_path()
        self._cleanup_stale_socket(sock_path)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(sock_path)
        self._sock.listen(5)
        self._sock.settimeout(1.0)  # 允许定期检查 _running 标志
        self._running = True

        # 信号处理
        def _shutdown(signum, frame):
            print("\n  收到停止信号，正在关闭...", file=sys.stderr, flush=True)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # 启动 idle watcher 线程
        watcher = threading.Thread(target=self._idle_watcher, daemon=True)
        watcher.start()

        print(f"Voxy 守护进程已启动，监听: {sock_path}", file=sys.stderr, flush=True)
        print(f"  STT 后端: {self._config.stt.backend}", file=sys.stderr, flush=True)
        print(f"  空闲超时: {self._config.daemon.idle_timeout} 分钟", file=sys.stderr, flush=True)

        try:
            while self._running:
                try:
                    conn, _ = self._sock.accept()
                    self._handle_connection(conn)
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        raise
                    break
        finally:
            self._sock.close()
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            self._unload_engine()
            print("Voxy 守护进程已停止", file=sys.stderr, flush=True)
