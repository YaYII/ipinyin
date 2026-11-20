"""朗读：把当前候选或刚上屏的文字念出来（粤语模式念粤语）。

两级后端，都是子进程调用、不链接、不复制代码，本项目仍是 MIT：

1. edge-tts（可选，GPLv3）：微软神经声线，粤语声调准确、接近真人；合成要联网，
   失败就退回 espeak。装上才用，没装不影响任何功能。
2. espeak-ng（系统包，GPLv3）：纯本地兜底，机器人音，粤语六声调实测不准。

合成与播放都放在后台线程，IBus 主循环发一个指令就返回。
两个后端都没有时静默降级：朗读不可用，输入法其他一切照常。
"""
from __future__ import annotations

import array
import collections
import hashlib
import logging
import marshal
import os
import shutil
import subprocess
import tempfile
import threading
import time

from . import config, tts_metrics, units

LOG = logging.getLogger("ipinyin")

DEFAULT_SPEED = 165      # 比 espeak 默认稍慢：粤语六个声调，听清楚比听快重要
MAX_CHARS = 0            # 0 = 不限制：有多少念多少
CACHE_LIMIT = 200        # 缓存 wav 条数上限
BATCH_WAIT = 0.04        # 取批时等 40ms，让连打进来的字并进同一句（人打字间隔 >40ms 就各自播）
CACHE_VERSION = "2"      # 裁静音后旧缓存（短句 1.87s）作废，键里带版本号

# 普通话声线读汉字会逐字拼英文字母（们→m-e-n-f-a-i-v），必须喂拼音；
# 粤语声线字典完整，简繁汉字都认，直接给汉字。
PINYIN_VOICE = "cmn-Latn-pinyin"
YUE_VOICE = "yue"

# 输出模式 → 声线
# espeak 没有潮州话声线：潮拼借用粤拼声线读（同属粤闽音节结构，近似但不等于真人）
CHAO_VOICE = "yue-Latn-jyutping"
VOICE_BY_MODE = {"simp": PINYIN_VOICE, "trad": PINYIN_VOICE, "yue": YUE_VOICE,
                 "chao": CHAO_VOICE}

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
    # 不截断：句子有多长就念多长
    return " ".join(cleaned.split())


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


def _fade(pcm: bytes, sr: int, ms: int = 6) -> bytes:
    """首尾各 6ms 淡入淡出。

    裁静音是从"第一个超过阈值的样本"开始切，切点几乎不可能正好在过零点上，
    播放时就是一个 DC 台阶——听起来就是每句开头"嗒"一声（用户说的按钮音）。
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


def _trim_wav(data: bytes) -> bytes:
    """裁掉 edge 补的静音。

    edge 对短文本会把音频补到 1.87s（「聽日」两个字也 1.87s），不裁的话
    两个字的词听起来像两秒，远低于正常语速（粤语连续语流约 4–6 音节/秒）。
    """
    if len(data) <= 44:
        return data
    pcm = units._trim(data[44:])
    if not pcm:
        return data
    pcm = _fade(pcm, units.SR)
    return units._wav_header(len(pcm)) + pcm


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
    """按队列逐条念：连打不丢音，一条念完接着下一条。

    以前是"新的打断旧的"（代次机制），打字快的时候前面的音全被吞掉，
    听感就是"有延迟而且缺字"。现在排队，只有显式 stop() 才清队。
    """

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
        self._cv = threading.Condition(self._lock)
        self._queue: collections.deque = collections.deque()
        self._prefetch: dict = {}      # (text, voice) -> 已合成好的结果，边播边合成下一条
        self._warming: set = set()     # 正在后台预合成的文本，避免重复请求
        self._worker: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._epoch = 0      # 只有 stop() 会推进；推进后队列里的旧条目作废
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
        offline = (os.environ.get("IPINYIN_TTS_OFFLINE", "0") == "1"
                   or not config.tts_edge_enabled())     # 状态文件可运行时切换
        # 本地音库优先：命中就 30-200ms 出声；没命中才联网，联网结果再收进音库
        if self._units is not None and voice == YUE_VOICE:
            backends.append("units")
        if self._edge and not offline:
            backends.append("edge")
        if self._binary:
            backends.append("espeak")
        for backend in backends:
            path = self._cache_path(text, voice, backend)
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
                seconds = max(len(data) - 44, 0) / 2 / units.SR
                tts_metrics.record(text, "cache", 0, seconds)   # 命中率也要看得见
                return data, path
            except OSError:
                pass
            started = time.time()
            if backend == "units":
                data = self._units.render(text)
            elif backend == "edge":
                data = self._edge_wav(text, voice)
                # 在线合成成功就顺手收进本地音库：下次同一個词离线可播
                if data and voice == YUE_VOICE and self._units is not None:
                    try:
                        self._units.add(text, data)
                    except Exception:
                        pass
            else:
                data = self._espeak_wav(text, voice)
            if data is None:
                continue
            self._store(path, data)
            seconds = max(len(data) - 44, 0) / 2 / units.SR
            tts_metrics.record(text, backend, (time.time() - started) * 1000, seconds)
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
                    return _trim_wav(handle.read())
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
        with self._cv:
            self._queue.append((text, voice))
            self._cv.notify()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="tts", daemon=True)
                self._worker.start()
        return True

    def warm(self, text: str, voice: str = PINYIN_VOICE) -> None:
        """后台预合成，不播放。用户上屏前先把候选合成好，上屏就能秒出。"""
        if not self.available:
            return
        text = normalize(text)
        if not text:
            return
        with self._cv:
            if text in self._warming or (text, voice) in self._prefetch:
                return
            self._warming.add(text)
        def work():
            try:
                self._render(text, voice)      # 命中缓存就是空操作
            except Exception:
                pass
            finally:
                with self._cv:
                    self._warming.discard(text)
        threading.Thread(target=work, name="tts-warm", daemon=True).start()

    def stop(self) -> None:
        """清空队列并打断正在念的那条（显式中断，例如切模式/退出）。"""
        with self._cv:
            self._queue.clear()
            self._prefetch.clear()
            self._epoch += 1
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _take_batch(self) -> tuple[str, str, int]:
        """取一批：队首 + 队列里同声线的后续条目，合并成一句。

        edge 每条合成要 1.1s；积压 4 条就是 4.4s。合并成一句只合成一次，
        内容一个字都不少，只是连着念。
        """
        with self._cv:
            while not self._queue:
                self._cv.wait()
            text, voice = self._queue.popleft()
            parts = [text]
            deadline = time.monotonic() + BATCH_WAIT
            while True:
                while (self._queue and self._queue[0][1] == voice
                       and (MAX_CHARS <= 0
                            or len("".join(parts)) + len(self._queue[0][0]) <= MAX_CHARS)):
                    parts.append(self._queue.popleft()[0])
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)      # 等 40ms 看还有没有字进来
            return "".join(parts), voice, self._epoch

    def _run(self) -> None:
        """唯一的朗读线程：取一批、念一批、念完取下一批。"""
        while True:
            text, voice, epoch = self._take_batch()

            try:
                self._say(text, voice, epoch)
            except Exception:          # 朗读永远不能把主循环带崩
                LOG.exception("朗读线程异常")

    def _say(self, text: str, voice: str, epoch: int) -> None:
        got = self._prefetch.pop((text, voice), None) or self._render(text, voice)
        if not got:
            return
        with self._cv:
            if epoch != self._epoch:      # stop() 过了，这条作废
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
        with self._cv:
            if epoch != self._epoch:      # 起播前最后一刻再确认一次
                proc.terminate()
                return
            self._proc = proc
        # 播放期间把下一条先合成好：把 1.1s 的合成藏进 0.6s 的播放里
        with self._cv:
            nxt = self._queue[0] if self._queue else None
            if len(self._prefetch) > 4:
                self._prefetch.clear()
        if nxt is not None and nxt not in self._prefetch:
            try:
                done = self._render(*nxt)
                if done:
                    self._prefetch[nxt] = done
            except Exception:          # 预取失败不影响当前播放
                pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            with self._cv:
                if self._proc is proc:
                    self._proc = None

    # ---------- 内部 ----------
    def _cache_path(self, text: str, voice: str, backend: str = "espeak") -> str:
        key = hashlib.sha1(
            f"{CACHE_VERSION}|{backend}|{voice}|{self._speed}|{text}".encode("utf-8")
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
