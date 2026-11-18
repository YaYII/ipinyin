"""朗读：把当前候选或刚上屏的文字念出来（粤语模式念粤语）。

两级后端，都是子进程调用、不链接、不复制代码，本项目仍是 MIT：

1. edge-tts（可选，GPLv3）：微软神经声线，粤语声调准确、接近真人；合成要联网，
   失败就退回 espeak。装上才用，没装不影响任何功能。
2. espeak-ng（系统包，GPLv3）：纯本地兜底，机器人音，粤语六声调实测不准。

合成与播放都放在后台线程，IBus 主循环发一个指令就返回。
两个后端都没有时静默降级：朗读不可用，输入法其他一切照常。
"""
from __future__ import annotations

import hashlib
import logging
import marshal
import os
import shutil
import subprocess
import tempfile
import threading

from . import config, units

LOG = logging.getLogger("ipinyin")

DEFAULT_SPEED = 165      # 比 espeak 默认稍慢：粤语六个声调，听清楚比听快重要
MAX_CHARS = 80           # 再长就别念了——一句话念十秒没人受得了
CACHE_LIMIT = 200        # 缓存 wav 条数上限

# 普通话声线读汉字会逐字拼英文字母（们→m-e-n-f-a-i-v），必须喂拼音；
# 粤语声线字典完整，简繁汉字都认，直接给汉字。
PINYIN_VOICE = "cmn-Latn-pinyin"
YUE_VOICE = "yue"

# 输出模式 → 声线
VOICE_BY_MODE = {"simp": PINYIN_VOICE, "trad": PINYIN_VOICE, "yue": YUE_VOICE}

# 按优先级找播放器：pipewire/pulse → alsa → ffmpeg
PLAYERS = (
    ("paplay", ("paplay",)),
    ("aplay", ("aplay", "-q")),
    ("ffplay", ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet")),
)

# 神经网络声线（可选）。espeak 的粤语六声调实测与调值不符
# （si1 该高平却下滑 50Hz、si3 该中平却上扬 42Hz），装上 edge-tts 就优先用它：
# 粤语走香港男声，简繁走普通话男声；失败（离线/超时）自动回落 espeak。
EDGE_VOICES = {YUE_VOICE: "zh-HK-WanLungNeural", PINYIN_VOICE: "zh-CN-YunxiNeural"}
EDGE_RATE = "+0%"
EDGE_TIMEOUT = 15   # 单句合成上限：网络卡住时宁可回落 espeak，也不能占着朗读线程


def voice_for(mode: str) -> str:
    """输出模式 → espeak 声线。"""
    return VOICE_BY_MODE.get(mode, PINYIN_VOICE)


def target(mode: str, word: str | None, last: str | None) -> tuple[str, str] | None:
    """念什么、用什么声线：优先当前候选，其次刚上屏那句。

    声线只由模式决定——念上屏句时也必须跟着模式走，
    否则粤语模式下会拿普通话音去读「嘅 / 唔啱」，听起来完全是另一个字。
    """
    text = word or last
    if not text:
        return None
    return text, voice_for(mode)


def normalize(text: str) -> str:
    """压掉控制字符与多余空白，过长截断。

    标点保留——espeak 会把它们念成自然停顿，去掉反而不像话。
    """
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    return " ".join(cleaned.split())[:MAX_CHARS]


def to_pinyin(text: str, table: dict[str, str]) -> str:
    """汉字 → 带调号拼音，喂给 espeak 的 cmn-Latn-pinyin 声线。

    espeak 的普通话声线读汉字会逐字拼字母（`们`→`m-e-n-f-a-i-v`），
    读拼音才是对的。ASCII 字母数字留着（API、v2、2026 这些），
    其余全角标点/emoji 丢掉——念出来只会是噪音。
    """
    out: list[str] = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff" or char == "〇":
            reading = table.get(char)
            if reading:
                out.append(" " + reading)   # 每个汉字一个音节，音节之间必须有空格
        elif char.isascii() and (char.isalnum() or char in " .,!?;:'-"):
            out.append(char)
    return "".join(out).strip()


_CHARS: dict[str, str] | None = None


def _char_pinyin() -> dict[str, str]:
    """懒加载汉字读音表；缺文件时返回空表（朗读会退回原样文本）。"""
    global _CHARS
    if _CHARS is None:
        try:
            with open(config.chars_path(), "rb") as handle:
                _CHARS = marshal.load(handle)
        except (OSError, ValueError, EOFError):
            LOG.warning("缺汉字读音表，普通话朗读会退回原样文本（跑 tools/build_char_pinyin.py）")
            _CHARS = {}
    return _CHARS


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


def _find_units() -> units.UnitLibrary | None:
    """词级音库（tools/build_tts_units.py 预生成）；没建库或关掉就返回 None。

    只在粤语声线用：库里的单元是粤语男声，拿去念普通话会是另一种语言。
    """
    if os.environ.get("IPINYIN_TTS_UNITS", "1") == "0":
        return None
    library = units.UnitLibrary(units.default_root())
    return library if library.available else None


def _find_player() -> list[str] | None:
    for name, argv in PLAYERS:
        if shutil.which(name):
            return list(argv)
    return None


def _find_edge() -> list[str] | None:
    """找 edge-tts：PATH → 项目私有 lib → 免 root 安装位 → 系统 python 模块。

    项目私有 lib（~/.local/share/ipinyin/lib）是免 root 位置：Ubuntu 24.04 的
    PEP 668 会拒绝 `pip install --user`，所以依赖装到项目目录、用 -c 注入 sys.path。
    """
    if not config.tts_edge_enabled():
        return None
    found = shutil.which("edge-tts")
    if found:
        return [found]
    for base in ("~/.local/share/ipinyin/bin", "~/.local/bin"):
        candidate = os.path.expanduser(os.path.join(base, "edge-tts"))
        if os.access(candidate, os.X_OK):
            return [candidate]
    python = shutil.which("python3") or shutil.which("python")
    lib = os.path.expanduser("~/.local/share/ipinyin/lib")
    if python and os.path.isdir(os.path.join(lib, "edge_tts")):
        code = (
            "import sys; sys.path.insert(0, %r); "
            "from edge_tts.util import main; main()" % lib
        )
        return [python, "-c", code]
    return None


class Speaker:
    """一次只念一条：新的朗读先打断上一条，不叠音。"""

    def __init__(self, binary=None, player=None, cache_dir=None, speed: int = DEFAULT_SPEED,
                 edge: list[str] | None | str = "auto",
                 unit_lib: units.UnitLibrary | None | str = "auto"):
        self._binary = binary if binary is not None else _find_espeak()
        # "auto" = 自己探测；None = 明确不要 edge（测试用）；列表 = 指定命令
        self._edge = _find_edge() if edge == "auto" else (list(edge) if edge else None)
        self._units = _find_units() if unit_lib == "auto" else unit_lib
        self._player = list(player) if player is not None else _find_player()
        self._cache_dir = cache_dir or config.tts_cache_dir()
        self._speed = speed
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._gen = 0        # 朗读代次：慢合成（edge 要 1–5s）期间连按，只有最后一条能出声
        self._warned = False

    @property
    def available(self) -> bool:
        return bool((self._units or self._edge or self._binary) and self._player)

    # ---------- 合成 ----------
    def wav(self, text: str, voice: str = PINYIN_VOICE) -> bytes | None:
        """合成一段 wav；命中缓存就不再调引擎。失败返回 None（绝不抛）。"""
        got = self._render(text, voice)
        return got[0] if got else None

    def _render(self, text: str, voice: str) -> tuple[bytes, str] | None:
        """按 edge → espeak 的顺序合成，返回 (音频, 缓存路径)。

        缓存键带 backend：同一句话两种引擎产出的音频不能互相冒充，
        否则装了 edge 之后会一直播到旧的 espeak 机械音。
        """
        text = normalize(text)
        if not text:
            return None
        backends = []
        offline = os.environ.get("IPINYIN_TTS_OFFLINE", "0") == "1"
        if self._edge and not offline:
            # 整句合成韵律自然；结果进缓存，同一句第二次起离线也能播
            backends.append("edge")
        if self._units is not None and voice == YUE_VOICE:
            # 离线兜底：词级拼接，每个音固定、可逐条校对，但词间有边界
            backends.append("units")
        if self._binary:
            backends.append("espeak")
        for backend in backends:
            path = self._cache_path(text, voice, backend)
            try:
                with open(path, "rb") as handle:
                    return handle.read(), path
            except OSError:
                pass
            if backend == "units":
                data = self._units.render(text)
            elif backend == "edge":
                data = self._edge_wav(text, voice)
            else:
                data = self._espeak_wav(text, voice)
            if data is None:
                continue
            self._store(path, data)
            return data, path
        return None

    def _espeak_wav(self, text: str, voice: str) -> bytes | None:
        spoken = to_pinyin(text, _char_pinyin()) if voice == PINYIN_VOICE else text
        if not spoken:
            return None
        try:
            done = subprocess.run(
                [self._binary, "-v", voice, "-s", str(self._speed), "--stdout", "--", spoken],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("朗读合成失败（espeak）：%s", exc)
            return None
        if done.returncode != 0 or not done.stdout:
            LOG.warning("朗读合成失败（espeak）：退出码 %s", done.returncode)
            return None
        return done.stdout

    def _edge_wav(self, text: str, voice: str) -> bytes | None:
        """edge-tts 合成 → ffmpeg 转 wav；任何一步失败都返回 None 让上层回落。"""
        edge_voice = EDGE_VOICES.get(voice)
        ffmpeg = shutil.which("ffmpeg")
        if not edge_voice or not self._edge or not ffmpeg:
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="ipinyin-tts-") as tmp:
                mp3 = os.path.join(tmp, "a.mp3")
                wav = os.path.join(tmp, "a.wav")
                done = subprocess.run(
                    [*self._edge, "--voice", edge_voice, "--rate", EDGE_RATE,
                     "--text", text, "--write-media", mp3],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=EDGE_TIMEOUT,
                    check=False,
                )
                if done.returncode != 0 or not os.path.exists(mp3):
                    LOG.warning("朗读合成失败（edge）：退出码 %s，回落 espeak", done.returncode)
                    return None
                conv = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", mp3,
                     "-ar", "22050", "-ac", "1", wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=EDGE_TIMEOUT,
                    check=False,
                )
                if conv.returncode != 0 or not os.path.exists(wav):
                    LOG.warning("朗读转码失败（ffmpeg）：退出码 %s，回落 espeak", conv.returncode)
                    return None
                with open(wav, "rb") as handle:
                    return handle.read()
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("朗读合成失败（edge）：%s，回落 espeak", exc)
            return None

    # ---------- 播放 ----------
    def speak(self, text: str, voice: str = PINYIN_VOICE) -> bool:
        """念一句。立刻返回，合成与播放在后台线程里做。"""
        if not self.available:
            if not self._warned:
                self._warned = True
                LOG.warning("朗读不可用：缺 espeak-ng / edge-tts / 音频播放器"
                            "（apt install espeak-ng 或 pip install --user edge-tts）")
            return False
        text = normalize(text)
        if not text:
            return False
        self.stop()
        with self._lock:
            self._gen += 1
            gen = self._gen
        threading.Thread(target=self._play, args=(text, voice, gen), name="tts", daemon=True).start()
        return True

    def stop(self) -> None:
        """打断正在念的那条（含还在合成的：代次一变，旧线程合成完也不许出声）。"""
        with self._lock:
            self._gen += 1
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _play(self, text: str, voice: str, gen: int) -> None:
        got = self._render(text, voice)
        if not got:
            return
        with self._lock:
            if gen != self._gen:      # 合成期间用户又按了（或 stop），这条作废
                return
        data, path = got
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
            if gen != self._gen:      # 起播前最后一刻再确认一次，避免叠音
                proc.terminate()
                return
            self._proc = proc
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---------- 内部 ----------
    def _cache_path(self, text: str, voice: str, backend: str = "espeak") -> str:
        key = hashlib.sha1(
            f"{backend}|{voice}|{self._speed}|{text}".encode("utf-8")
        ).hexdigest()[:16]
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
