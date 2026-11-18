"""词级音库：把「词 → 预合成音频」存成本地单元，朗读时按词拼接，完全离线。

单元由 `tools/build_tts_units.py` 用 edge-tts 预生成一次（Opus 24kbps），
之后朗读不再联网，也不再看 edge 的脸色。拼接方式是「词单元 + 40ms 间隙」，
20 句 A/B 对比实测与整句合成听感接近。

覆盖不全（生僻字/新词）时 `render()` 返回 None，由上层回落到 edge / espeak。
"""
from __future__ import annotations

import array
import json
import logging
import os
import shutil
import struct
import subprocess
import threading

LOG = logging.getLogger("ipinyin")

SR = 22050          # 单元统一采样率：edge 输出是 24kHz，转成 22.05k 体积小一点
GAP_MS = 15         # 词间静音：40ms 听起来一字一顿，收到 15ms 才连贯
CACHE_LIMIT = 400   # 解码后的 PCM 缓存条数（约 400 × 1s × 44KB ≈ 17MB）
SILENCE = 300       # 静音阈值（16bit 幅度）：edge 对短文本会补到 1.87s，必须裁掉
HEAD_MS = 5         # 裁完留一点头，避免起音被切
TAIL_MS = 25        # 留一点尾，避免尾音被切；太长会吃掉连读


def default_root() -> str:
    """音库目录：默认在用户数据目录，可用 IPINYIN_TTS_UNITS_DIR 覆盖。

    路径不放进 config.py——那个文件在并行改动里（个人词频分家），避免撞车。
    """
    return os.environ.get("IPINYIN_TTS_UNITS_DIR") or os.path.expanduser(
        "~/.local/share/ipinyin/tts-units"
    )


def _wav_header(n: int, sr: int = SR) -> bytes:
    """44 字节标准 wav 头（PCM 16bit 单声道）。"""
    return (
        b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", n)
    )


def _trim(pcm: bytes, sr: int = SR) -> bytes:
    """裁掉首尾静音：edge 会把短文本补到 1.87s，不裁就是每个词后面停 1.3 秒。"""
    if len(pcm) < 2:
        return b""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    start = 0
    while start < len(samples) and abs(samples[start]) < SILENCE:
        start += 1
    end = len(samples)
    while end > start and abs(samples[end - 1]) < SILENCE:
        end -= 1
    if end <= start:
        return b""
    head = int(sr * HEAD_MS / 1000)
    tail = int(sr * TAIL_MS / 1000)
    start = max(0, start - head)
    end = min(len(samples), end + tail)
    return samples[start:end].tobytes()


class UnitLibrary:
    """词 → Opus 单元 → 拼接成 wav。线程安全（朗读在后台线程）。"""

    def __init__(self, root: str, ffmpeg: str | None = None):
        self._root = root
        self._index: dict[str, str] = {}
        self._maxlen = 1
        self._pcm: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._load()

    def _load(self) -> None:
        try:
            with open(os.path.join(self._root, "index.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict):
            self._index = {str(k): str(v) for k, v in data.items() if k and v}
        self._maxlen = max((len(k) for k in self._index), default=1)

    @property
    def available(self) -> bool:
        return bool(self._index and self._ffmpeg)

    def __len__(self) -> int:
        return len(self._index)

    def segment(self, text: str) -> list[str] | None:
        """最长匹配切词；只要有一个字/词不在库里就返回 None（整句回落）。"""
        if not self._index:
            return None
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if not ("\u4e00" <= ch <= "\u9fff"):
                if ch.isalnum():
                    return None      # 数字/字母拼接念不好，整句交给 edge
                i += 1               # 标点不出声，直接跳过
                continue
            for n in range(min(self._maxlen, len(text) - i), 0, -1):
                piece = text[i:i + n]
                if piece in self._index:
                    out.append(piece)
                    i += n
                    break
            else:
                return None
        return out

    def render(self, text: str) -> bytes | None:
        """整句 → wav bytes；覆盖不全或解码失败返回 None。"""
        pieces = self.segment(text)
        if not pieces:
            return None
        gap = b"\x00\x00" * int(SR * GAP_MS / 1000)
        chunks: list[bytes] = []
        for piece in pieces:
            pcm = self._decode(self._index[piece])
            if not pcm:
                return None
            chunks.append(pcm)
        body = gap.join(chunks) + gap
        return _wav_header(len(body)) + body

    def _decode(self, name: str) -> bytes | None:
        with self._lock:
            hit = self._pcm.get(name)
        if hit is not None:
            return hit
        try:
            done = subprocess.run(
                [self._ffmpeg, "-v", "error", "-i", os.path.join(self._root, name),
                 "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("音库单元解码失败：%s", exc)
            return None
        if done.returncode != 0 or not done.stdout:
            LOG.warning("音库单元解码失败：%s 退出码 %s", name, done.returncode)
            return None
        done_stdout = _trim(done.stdout)
        if not done_stdout:
            return None
        done = subprocess.CompletedProcess(done.args, 0, done_stdout, b"")
        with self._lock:
            if len(self._pcm) >= CACHE_LIMIT:
                self._pcm.clear()
            self._pcm[name] = done.stdout
        return done.stdout
