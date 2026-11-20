"""词级音库：把「词 → 预合成音频」存成本地单元，朗读时按词拼接，完全离线。

单元由 `tools/build_tts_units.py` 用 edge-tts 预生成一次（Opus 24kbps），
之后朗读不再联网，也不再看 edge 的脸色。拼接方式是「词单元 + 40ms 间隙」，
20 句 A/B 对比实测与整句合成听感接近。

覆盖不全（生僻字/新词）时 `render()` 返回 None，由上层回落到 edge / espeak。
"""
from __future__ import annotations

import array
import atexit
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import threading

LOG = logging.getLogger("ipinyin")

SR = 22050          # 单元统一采样率：edge 输出是 24kHz，转成 22.05k 体积小一点
GAP_MS = 15
# 标点停顿：逗号短、句号长。以前标点被直接跳过，整句一口气念完，听感很赶。
PAUSE_MS = {
    "，": 180, ",": 150, "、": 120,
    "。": 360, ".": 300, "！": 320, "!": 300, "？": 320, "?": 300,
    "；": 220, ";": 200, "：": 160, ":": 150,
}         # 词间静音：40ms 听起来一字一顿，收到 15ms 才连贯
CACHE_LIMIT = 400   # 解码后的 PCM 缓存条数（约 400 × 1s × 44KB ≈ 17MB）
SILENCE = 300       # 静音阈值（16bit 幅度）：edge 对短文本会补到 1.87s，必须裁掉
HEAD_MS = 5         # 裁完留一点头，避免起音被切
TAIL_MS = 25        # 留一点尾，避免尾音被切；太长会吃掉连读


_SYL2CHAR: dict | None = None
_YUE_CONV = None


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


def _fade(pcm: bytes, sr: int = SR, ms: int = 4) -> bytes:
    """每个单元首尾 4ms 淡入淡出。

    裁静音是从非零样本切开的，直接拼会在每个字边界留一个台阶——听感就是
    「磨砂 / 不融合」。淡入淡出把台阶抹掉，代价是丢 4ms 起音。
    """
    n = len(pcm) // 2
    if n < 8:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: n * 2])
    f = min(int(sr * ms / 1000), n // 2)
    for i in range(f):
        g = i / f
        samples[i] = int(samples[i] * g)
        samples[n - 1 - i] = int(samples[n - 1 - i] * g)
    return samples.tobytes()


def _join(chunks: list[bytes], sr: int = SR, fade_ms: int = 20) -> bytes:
    """单元拼接：相邻两段做 20ms 交叉淡化，而不是插静音。

    插 15ms 静音会让每个字边界出现一个谱通量尖峰（实测 8 个 vs 整句合成 2 个），
    听感就是「一字一顿」。交叉淡化把边界抹平，还顺带把句子缩短约 20%。
    """
    if not chunks:
        return b""
    body = chunks[0]
    for nxt in chunks[1:]:
        n = min(int(sr * fade_ms / 1000), len(body) // 2, len(nxt) // 2)
        if n <= 0:
            body += nxt
            continue
        a = array.array("h")
        a.frombytes(body[-n * 2:])
        b = array.array("h")
        b.frombytes(nxt[: n * 2])
        mixed = array.array("h", (int(a[i] * (1 - i / n) + b[i] * (i / n))
                                  for i in range(n)))
        body = body[: -n * 2] + mixed.tobytes() + nxt[n * 2:]
    return body


def _split_clauses(text: str) -> list[tuple[str, int]]:
    """按标点切成分句，返回 [(分句, 停顿毫秒)]；最后一组停顿为 0。"""
    out: list[tuple[str, int]] = []
    buf = ""
    for ch in text:
        if ch in PAUSE_MS:
            out.append((buf.strip(), PAUSE_MS[ch]))
            buf = ""
        else:
            buf += ch
    out.append((buf.strip(), 0))
    return out


class UnitLibrary:
    """词 → Opus 单元 → 拼接成 wav。线程安全（朗读在后台线程）。"""

    def __init__(self, root: str, ffmpeg: str | None = None):
        self._root = root
        self._index: dict[str, str] = {}
        self._dirty = 0
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
        atexit.register(self.flush)      # 进程退出前把攒的单元写盘

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
        """整句 → wav bytes；标点处插入停顿；词/字都查不到时退回按音节拼接。"""
        clauses = _split_clauses(text)
        if len(clauses) > 1:
            body = b""
            for clause, pause in clauses:
                if clause:
                    pcm = self._render_pcm(clause)
                    if pcm is None:
                        return None
                    body += pcm
                if pause:
                    body += b"\x00\x00" * int(SR * pause / 1000)
            return _wav_header(len(body)) + body
        pcm = self._render_pcm(text)
        return None if pcm is None else _wav_header(len(pcm)) + pcm

    def _render_pcm(self, text: str) -> bytes | None:
        """一段没有标点的文字 → PCM。"""
        pieces = self.segment(text)
        if pieces is None:
            return self._render_by_syllable(text)
        if not pieces:
            return None
        chunks = self._decode_all([self._index[p] for p in pieces])
        if chunks is None:
            return None
        return _join(chunks)

    def _decode_all(self, names: list[str]) -> list[bytes] | None:
        """并发解码一批单元。

        每个单元一次 ffmpeg 调用（~110ms）；42 字串行就是 4.6 秒，用户等不起。
        并发 8 路把首播延迟压到 0.6 秒内，结果仍进 _pcm 缓存，重播是 0 成本。
        """
        todo = list(dict.fromkeys(n for n in names if n not in self._pcm))
        if len(todo) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                list(pool.map(self._decode, todo))
        out: list[bytes] = []
        for name in names:
            pcm = self._pcm.get(name)
            if pcm is None:
                pcm = self._decode(name)
            if not pcm:
                return None
            out.append(pcm)
        return out

    def _render_by_syllable(self, text: str) -> bytes | None:
        """B 级兜底：汉字 → 粤拼音节 → 代表字单元 → 拼接。

        词级库再全也有查不到的词；音节级是有限集（1967 个），拼得出来就念得出来。
        代表字是「读音唯一、尽量常用」的那个字，单元本身仍是 edge 录的真音。
        """
        global _SYL2CHAR, _YUE_CONV
        if _SYL2CHAR is None:
            try:
                import json as _json
                with open(os.path.join(self._root, "syllables.json"), encoding="utf-8") as fh:
                    _SYL2CHAR = _json.load(fh)
            except (OSError, ValueError):
                _SYL2CHAR = {}
        if not _SYL2CHAR:
            return None
        if _YUE_CONV is None:
            try:
                from . import config, yue
                _YUE_CONV = yue.YueConverter.load_dict(config.yue_dict_path())
            except Exception:
                _YUE_CONV = False
        if not _YUE_CONV:
            return None
        reading = _YUE_CONV.jyutping(text)
        if not reading:
            return None
        names: list[str] = []
        for syl in reading.split():
            ch = _SYL2CHAR.get(syl)
            if not ch or ch not in self._index:
                return None
            names.append(self._index[ch])
        chunks = self._decode_all(names)
        if chunks is None:
            return None
        if not chunks:
            return None
        body = _join(chunks)
        return _wav_header(len(body)) + body

    def add(self, text: str, wav: bytes) -> bool:
        """把一段在线合成好的音频收进音库：用得越多，越少联网。

        只收短词（≤4 字）：整句存进来会污染音库，而且拼接时用不上。
        索引攒够一批再落盘，避免每次上屏都重写 index.json。
        """
        if not text or len(text) > 4 or len(wav) <= 44:
            return False
        if text in self._index:
            return False
        if not self._ffmpeg:
            return False
        name = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] + ".opus"
        target = os.path.join(self._root, name)
        src = target + ".tmp.wav"
        try:
            with open(src, "wb") as fh:
                fh.write(wav)
            done = subprocess.run(
                [self._ffmpeg, "-y", "-v", "error", "-i", src, "-c:a", "libopus",
                 "-b:a", "24k", "-ar", "24000", "-ac", "1", target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            if done.returncode != 0 or not os.path.exists(target):
                return False
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            if os.path.exists(src):
                try:
                    os.unlink(src)
                except OSError:
                    pass
        with self._lock:
            self._index[text] = name
            self._maxlen = max(self._maxlen, len(text))
            self._dirty += 1
            flush = self._dirty >= 10
        if flush:
            self.flush()
        return True

    def flush(self) -> None:
        """把新增单元写进 index.json（原子替换）。"""
        with self._lock:
            if self._dirty <= 0:
                return
            snapshot = dict(self._index)
            self._dirty = 0
        path = os.path.join(self._root, "index.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False, indent=0, sort_keys=True)
            os.replace(tmp, path)
        except OSError:
            pass

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
        done_stdout = _fade(_trim(done.stdout))
        if not done_stdout:
            return None
        done = subprocess.CompletedProcess(done.args, 0, done_stdout, b"")
        with self._lock:
            if len(self._pcm) >= CACHE_LIMIT:
                self._pcm.clear()
            self._pcm[name] = done.stdout
        return done.stdout
