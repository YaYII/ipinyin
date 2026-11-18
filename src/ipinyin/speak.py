"""朗读：把当前候选或刚上屏的文字念出来（粤语模式念粤语）。

后端是 espeak-ng（系统包，GPLv3）——只以子进程方式调用，不链接、不复制其代码，
本项目仍是 MIT。合成与播放都放在后台线程，IBus 主循环发一个指令就返回。

没装 espeak-ng 时静默降级：朗读不可用，输入法其他一切照常。
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading

from . import config

LOG = logging.getLogger("ipinyin")

DEFAULT_SPEED = 165      # 比 espeak 默认稍慢：粤语六个声调，听清楚比听快重要
MAX_CHARS = 80           # 再长就别念了——一句话念十秒没人受得了
CACHE_LIMIT = 200        # 缓存 wav 条数上限

# 输出模式 → 声线。粤语模式念粤语；简繁都念普通话（繁体字 espeak 也认）
VOICE_BY_MODE = {"simp": "cmn", "trad": "cmn", "yue": "yue"}

# 按优先级找播放器：pipewire/pulse → alsa → ffmpeg
PLAYERS = (
    ("paplay", ("paplay",)),
    ("aplay", ("aplay", "-q")),
    ("ffplay", ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet")),
)


def voice_for(mode: str) -> str:
    """输出模式 → espeak 声线。"""
    return VOICE_BY_MODE.get(mode, "cmn")


def normalize(text: str) -> str:
    """压掉控制字符与多余空白，过长截断。

    标点保留——espeak 会把它们念成自然停顿，去掉反而不像话。
    """
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    return " ".join(cleaned.split())[:MAX_CHARS]


def _find_espeak() -> str | None:
    """找 espeak-ng：先 PATH，再看免 root 的本地安装位。

    Ubuntu 只装 libespeak-ng1 时数据是齐的（含 yue_dict），缺的只是命令行那个二进制；
    把 espeak-ng 放进 ~/.local/bin 就能用，不必 sudo。
    """
    found = shutil.which("espeak-ng")
    if found:
        return found
    for base in ("~/.local/bin", "~/.local/share/ipinyin/bin"):
        candidate = os.path.expanduser(os.path.join(base, "espeak-ng"))
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_player() -> list[str] | None:
    for name, argv in PLAYERS:
        if shutil.which(name):
            return list(argv)
    return None


class Speaker:
    """一次只念一条：新的朗读先打断上一条，不叠音。"""

    def __init__(self, binary=None, player=None, cache_dir=None, speed: int = DEFAULT_SPEED):
        self._binary = binary if binary is not None else _find_espeak()
        self._player = list(player) if player is not None else _find_player()
        self._cache_dir = cache_dir or config.tts_cache_dir()
        self._speed = speed
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._warned = False

    @property
    def available(self) -> bool:
        return bool(self._binary and self._player)

    # ---------- 合成 ----------
    def wav(self, text: str, voice: str = "cmn") -> bytes | None:
        """合成一段 wav；命中缓存就不再调 espeak。失败返回 None（绝不抛）。"""
        text = normalize(text)
        if not text or not self._binary:
            return None
        path = self._cache_path(text, voice)
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError:
            pass
        try:
            done = subprocess.run(
                [self._binary, "-v", voice, "-s", str(self._speed), "--stdout", "--", text],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("朗读合成失败：%s", exc)
            return None
        if done.returncode != 0 or not done.stdout:
            LOG.warning("朗读合成失败：espeak-ng 退出码 %s", done.returncode)
            return None
        self._store(path, done.stdout)
        return done.stdout

    # ---------- 播放 ----------
    def speak(self, text: str, voice: str = "cmn") -> bool:
        """念一句。立刻返回，合成与播放在后台线程里做。"""
        if not self.available:
            if not self._warned:
                self._warned = True
                LOG.warning("朗读不可用：缺 espeak-ng 或音频播放器（apt install espeak-ng）")
            return False
        text = normalize(text)
        if not text:
            return False
        self.stop()
        threading.Thread(target=self._play, args=(text, voice), name="tts", daemon=True).start()
        return True

    def stop(self) -> None:
        """打断正在念的那条。"""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _play(self, text: str, voice: str) -> None:
        data = self.wav(text, voice)
        if not data:
            return
        path = self._cache_path(text, voice)
        if not os.path.exists(path):
            path = self._spill(data)
            if not path:
                return
        try:
            proc = subprocess.Popen(
                [*self._player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            LOG.warning("朗读播放失败：%s", exc)
            return
        with self._lock:
            self._proc = proc
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---------- 内部 ----------
    def _cache_path(self, text: str, voice: str) -> str:
        key = hashlib.sha1(f"{voice}|{self._speed}|{text}".encode("utf-8")).hexdigest()[:16]
        return os.path.join(self._cache_dir, f"{key}.wav")

    def _store(self, path: str, data: bytes) -> None:
        """写缓存（先写临时文件再 rename，读到的永远是完整 wav）。"""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except OSError:
            return
        self._trim()

    def _spill(self, data: bytes) -> str | None:
        """缓存目录不可写时，退到临时文件播放。"""
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(data)
                return handle.name
        except OSError:
            return None

    def _trim(self) -> None:
        """缓存只留最近的 CACHE_LIMIT 条，避免无限长大。"""
        try:
            files = [
                os.path.join(self._cache_dir, name)
                for name in os.listdir(self._cache_dir)
                if name.endswith(".wav")
            ]
        except OSError:
            return
        if len(files) <= CACHE_LIMIT:
            return
        files.sort(key=lambda item: os.path.getmtime(item))
        for path in files[: len(files) - CACHE_LIMIT]:
            try:
                os.unlink(path)
            except OSError:
                pass


_SPEAKER: Speaker | None = None


def get_speaker() -> Speaker:
    """进程内单例：缓存与「打断上一条」的状态都在这里。"""
    global _SPEAKER
    if _SPEAKER is None:
        _SPEAKER = Speaker()
    return _SPEAKER
