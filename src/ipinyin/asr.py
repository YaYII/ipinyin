"""语音识别：SenseVoiceSmall（sherpa-onnx，本地 CPU）。

模型从 sherpa-onnx 官方包解压到 ~/.local/share/ipinyin/asr/，int8 版 229MB。
识别一句 300ms 左右（4 线程），够输入法用——说话 → 文字 → commit_text。
"""
from __future__ import annotations

import logging
import os
import signal as _signal
import subprocess
import tempfile
import threading
import time

from . import config

LOG = logging.getLogger("ipinyin")

MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
# 语言标签：模型自带，yue=粤语、zh=普通话、auto=自动
LANGUAGES = ("yue", "zh", "auto")


def asr_root() -> str:
    return os.path.join(config.data_home(), "asr")


def model_dir() -> str:
    """模型目录：优先固定名 sensevoice，退回官方解压出来的长名目录。"""
    base = asr_root()
    fixed = os.path.join(base, "sensevoice")
    if os.path.exists(os.path.join(fixed, "model.int8.onnx")):
        return fixed
    long_name = os.path.join(base, MODEL_NAME)
    if os.path.exists(os.path.join(long_name, "model.int8.onnx")):
        return long_name
    return ""


class Recognizer:
    """懒加载的 SenseVoice 识别器：没装模型时 available=False，调用方回退。"""

    def __init__(self, language: str = "yue", threads: int = 4):
        self._language = language if language in LANGUAGES else "yue"
        self._threads = threads
        self._rec = None
        self._failed = False

    def available(self) -> bool:
        return bool(model_dir()) and not self._failed

    def _load(self):
        if self._rec is not None or self._failed:
            return self._rec
        directory = model_dir()
        if not directory:
            self._failed = True
            return None
        try:
            import sherpa_onnx

            started = time.time()
            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=os.path.join(directory, "model.int8.onnx"),
                tokens=os.path.join(directory, "tokens.txt"),
                num_threads=self._threads,
                use_itn=True,
                language=self._language,
                debug=False,
            )
            LOG.info("语音模型已加载 %.1fs（%s）", time.time() - started, self._language)
        except Exception as exc:      # 缺包 / 模型损坏：让上层回退，不崩引擎
            LOG.warning("语音模型加载失败：%s", exc)
            self._failed = True
        return self._rec

    def recognize_file(self, path: str) -> str:
        """识别一个 WAV 文件；失败返回空串。"""
        rec = self._load()
        if rec is None:
            return ""
        try:
            import wave

            import numpy as np

            with wave.open(path) as handle:
                rate = handle.getframerate()
                samples = np.frombuffer(handle.readframes(handle.getnframes()),
                                        "<i2").astype("float32") / 32768.0
            return self.recognize_samples(samples, rate)
        except Exception as exc:
            LOG.warning("语音识别失败：%s", exc)
            return ""

    def recognize_samples(self, samples, rate: int) -> str:
        rec = self._load()
        if rec is None:
            return ""
        stream = rec.create_stream()
        stream.accept_waveform(rate, samples)
        rec.decode_stream(stream)
        return (stream.result.text or "").strip()


def list_devices() -> list[str]:
    """所有可录音设备（plughw:c,d），第一个通常就是内置麦。"""
    out = []
    try:
        done = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                              timeout=3, env=dict(os.environ, LC_ALL="C"))
        import re

        for card, device in re.findall(r"card (\d+):.*?device (\d+):", done.stdout, re.S):
            out.append(f"plughw:{card},{device}")
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def capture_device() -> str:
    """找录音设备。

    系统默认源经常指向输出节点（PipeWire 上见过），直接 arecord 录到全静音。
    这里自己解析 arecord -l 取第一块采集卡，环境变量 IPINYIN_MIC 可覆盖。
    """
    env = os.environ.get("IPINYIN_MIC")
    if env:
        return env
    try:
        done = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                              timeout=3, env=dict(os.environ, LC_ALL="C"))
        import re

        found = re.search(r"card (\d+):.*?device (\d+):", done.stdout, re.S)
        if found:
            return f"plughw:{found.group(1)},{found.group(2)}"
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _new_path() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="ipinyin-asr-"), "a.wav")


def _spawn(path: str, seconds: float | None = None) -> subprocess.Popen:
    command = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav"]
    device = capture_device()
    if device:
        command[1:1] = ["-D", device]
    if seconds:
        command[1:1] = ["-d", str(int(seconds))]
    return subprocess.Popen([*command, path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def record(seconds: float, path: str | None = None) -> str:
    """定长录音，返回 wav 路径。"""
    path = path or _new_path()
    proc = _spawn(path, seconds)
    proc.wait(timeout=seconds + 5)
    return path


def start_recording(path: str | None = None) -> tuple[str, subprocess.Popen]:
    """开始录音（录到 stop_recording 为止）。"""
    path = path or _new_path()
    return path, _spawn(path)


def stop_recording(path: str, proc: subprocess.Popen, timeout: float = 3.0) -> str:
    """停止录音：先 SIGINT 让 arecord 写完 WAV 头，超时才 kill。"""
    if proc.poll() is None:
        proc.send_signal(_signal.SIGINT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
    return path


class Listener(threading.Thread):
    """常听模式：一直读麦克风，检测到说话就攒，静下来就识别。

    F8 开一次开始听，再按一次停止。阈值按开机后 0.6 秒的底噪自适应，
    免得安静房间一直触发、嘈杂环境一句也听不到。
    """

    FRAME_MS = 100
    START_FRAMES = 3          # 连续 3 帧高于阈值 = 开始说话
    SILENCE_FRAMES = 7        # 连续 7 帧（0.7s）低于阈值 = 说完了
    MIN_SPEECH_MS = 300       # 太短不当一句话
    MAX_SPEECH_MS = 15000     # 超长截断，防止一直录

    def __init__(self, on_text, language: str = "yue", device: str = "", level=None):
        super().__init__(daemon=True, name="ipinyin-voice")
        self._on_text = on_text
        self._language = language
        self._device = device or capture_device()
        self._level = level
        self._stop = threading.Event()
        self._proc = None
        self._rec = None

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _open(self):
        command = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"]
        if self._device:
            command[1:1] = ["-D", self._device]
        self._proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        return self._proc.stdout

    def run(self) -> None:
        import numpy as np

        try:
            stream = self._open()
        except OSError as exc:
            LOG.warning("语音常听启动失败：%s", exc)
            return
        frame_bytes = 16000 * self.FRAME_MS // 1000 * 2
        noise = 0.0           # 底噪（前 6 帧平均）
        calibrated = 0
        speaking = False
        voiced_run = 0
        silent_run = 0
        buf: list = []
        if self._rec is None:
            self._rec = Recognizer(language=self._language)
        pending = b""
        try:
            while not self._stop.is_set():
                # 管道 read 不保证读满：短读要攒，攒够一帧再处理（以前当 EOF 直接退出了）
                chunk = stream.read(frame_bytes - len(pending))
                if not chunk:
                    break
                pending += chunk
                if len(pending) < frame_bytes:
                    continue
                frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                x = np.frombuffer(frame, "<i2").astype("float32") / 32768.0
                rms = float(np.sqrt((x ** 2).mean()))
                if calibrated < 6:
                    noise = max(noise, rms) if calibrated else rms
                    calibrated += 1
                    if calibrated == 6:
                        noise = max(noise, 0.004)
                        LOG.info("语音常听就绪：设备 %s，底噪 %.4f，阈值 %.4f",
                                 self._device or "默认", noise, noise * 3)
                    continue
                threshold = max(noise * 3, 0.012)
                if rms > threshold:
                    voiced_run += 1
                    silent_run = 0
                else:
                    silent_run += 1
                    voiced_run = 0
                if not speaking:
                    if voiced_run >= self.START_FRAMES:
                        speaking = True
                        buf = [x]
                        LOG.info("语音：检测到说话")
                    continue
                buf.append(x)
                frames = len(buf)
                if silent_run >= self.SILENCE_FRAMES:
                    speaking = False
                    if frames * self.FRAME_MS < self.MIN_SPEECH_MS:
                        buf = []
                        continue
                    self._deliver(buf)
                    buf = []
                elif frames * self.FRAME_MS >= self.MAX_SPEECH_MS:
                    speaking = False
                    self._deliver(buf)
                    buf = []
        except Exception:
            LOG.exception("语音常听线程异常")
        finally:
            proc, self._proc = self._proc, None
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _deliver(self, buf) -> None:
        import numpy as np

        try:
            samples = np.concatenate(buf)
            LOG.info("语音：识别 %.1fs 音频", len(samples) / 16000)
            text = self._rec.recognize_samples(samples, 16000)
            if text:
                self._on_text(text)
            else:
                LOG.info("语音：这段没识别出内容")
        except Exception:
            LOG.exception("语音识别失败")


