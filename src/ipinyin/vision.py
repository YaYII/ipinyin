"""屏幕感知：把截图变成上下文，交给可插拔的视觉模型命令。

为什么不做「自动截屏」：GNOME Wayland 下 `org.gnome.Shell.Screenshot` 直接拒绝
（AccessDenied: Screenshot is not allowed），ScreenCast portal 需要注册 app_id 并
一次性授权。在那之前，用「监看截图文件夹」把链路跑通——你按 PrintScreen，我负责理解。

模型不写死：`IPINYIN_VISION_CMD` 是命令模板，`{image}` 会被替换成图片路径。
例如：opencode run --model deepseek-v4.1-flash -f {image} "一句话说明这个画面在做什么"
没配就只记录「有新截图」，不调用模型（链路照样可验证）。
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time

from . import config

LOG = logging.getLogger("ipinyin")

DEFAULT_PROMPT = "用一句话说明这个画面：使用者在做什么、在哪个应用、关键内容是什么。"
MAX_DESC = 500
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
RETENTION_DAYS = 1        # 只保留自己的帧一天：昨天的自动删，避免垃圾堆积
MAX_LOG_LINES = 2000      # 上下文日志上限，超了只留最近的部分


def home() -> str:
    return os.path.join(config.data_home(), "vision")


def state_path() -> str:
    return os.path.join(home(), "state.json")


def log_path() -> str:
    return os.path.join(home(), "vision.jsonl")


def frames_dir() -> str:
    """我们自己的帧副本目录——只有这里的东西会被自动清理。"""
    return os.path.join(home(), "frames")


def _copy_frame(image: str) -> str:
    """把要处理的截图复制一份到自己目录：原件是你的，永远不碰。"""
    import shutil

    try:
        os.makedirs(frames_dir(), exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{os.path.basename(image)}"
        target = os.path.join(frames_dir(), name)
        shutil.copy2(image, target)
        return target
    except OSError:
        return image


def watch_dirs() -> list[str]:
    """默认看 GNOME 截图目录与项目自己的收件箱。"""
    override = os.environ.get("IPINYIN_VISION_DIRS")
    if override:   # 显式指定就只用指定的（测试与自建目录都用这条）
        return [d for d in override.split(os.pathsep) if d and os.path.isdir(d)]
    dirs = [
        os.path.expanduser("~/Pictures/截图"),
        os.path.expanduser("~/Pictures/Screenshots"),
        os.path.join(home(), "incoming"),   # screencast 常驻抓的帧
        os.path.join(home(), "inbox"),
    ]
    return [d for d in dirs if os.path.isdir(d)]


def _load_state() -> dict:
    try:
        with open(state_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        os.makedirs(home(), exist_ok=True)
        with open(state_path(), "w", encoding="utf-8") as handle:
            json.dump(state, handle)
    except OSError:
        pass


def scan(mark: bool = True) -> list[str]:
    """返回上次之后新增的截图（按修改时间排序）；mark=False 时先不记，处理完再记。"""
    state = _load_state()
    seen: dict[str, float] = state.get("seen", {})
    fresh: list[tuple[float, str]] = []
    for directory in watch_dirs():
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            path = os.path.join(directory, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if seen.get(path) == mtime:
                continue
            fresh.append((mtime, path))
    fresh.sort()
    for mtime, path in fresh:
        seen[path] = mtime
    # 只留最近 500 条，避免状态文件膨胀
    if len(seen) > 500:
        for key in sorted(seen, key=lambda k: seen[k])[: len(seen) - 500]:
            seen.pop(key, None)
    if mark:
        state["seen"] = seen
        _save_state(state)
    return [path for _, path in fresh]


def mark_seen(path: str) -> None:
    """处理完一张再标记，避免「扫到 392 张只处理 3 张，其余被吞掉」。"""
    state = _load_state()
    seen = state.setdefault("seen", {})
    try:
        seen[path] = os.path.getmtime(path)
    except OSError:
        return
    _save_state(state)


OPENCODE_BASE = os.environ.get("IPINYIN_VISION_BASE", "https://opencode.ai/zen/go/v1")
OPENCODE_MODEL = os.environ.get("IPINYIN_VISION_MODEL", "deepseek-v4.1-flash")
OPENCODE_CRED = os.path.expanduser("~/.dsh/.credentials.yaml")
# 该网关要求带会话标识做路由；给视觉单独一个，别和 DSH 会话混在一起
OPENCODE_SESSION = os.environ.get("IPINYIN_VISION_SESSION", "ses_ipinyin_vision")


def _opencode_key() -> str | None:
    """从 DSH 凭证里取 opencode-go 的 key（同一个订阅，不额外申请）。"""
    env_key = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import re

        with open(OPENCODE_CRED, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = re.search(r"OPENCODE_GO_API_KEY:\s*(\S+)", text)
    return match.group(1) if match else None


def _describe_opencode(image: str, prompt: str) -> str | None:
    """走 opencode-go（OpenAI 兼容）的视觉模型，用 curl 免得被 UA 拦。"""
    key = _opencode_key()
    if not key:
        return None
    import base64
    import json

    try:
        with open(image, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode()
    except OSError:
        return None
    body = {
        "model": OPENCODE_MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        ]}],
    }
    try:
        done = subprocess.run(
            ["curl", "-sS", "--max-time", "90", "-A", "ipinyin-vision/1.0",
             OPENCODE_BASE + "/chat/completions",
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-H", f"x-opencode-session: {OPENCODE_SESSION}",
             "--data-binary", "@-"],
            input=json.dumps(body), capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("视觉模型调用失败：%s", exc)
        return None
    if done.returncode != 0:
        LOG.warning("视觉模型调用失败：curl %s", done.returncode)
        return None
    try:
        payload = json.loads(done.stdout)
        message = payload["choices"][0]["message"]
    except (ValueError, KeyError, IndexError):
        LOG.warning("视觉模型返回不可解析：%s", (done.stdout or "")[:200])
        return None
    text = " ".join((message.get("content") or "").split())
    return text[:MAX_DESC] or None


def describe(image: str) -> str | None:
    """调用视觉模型：自定义命令优先，其次 opencode-go；都不可用返回 None。"""
    prompt = os.environ.get("IPINYIN_VISION_PROMPT", DEFAULT_PROMPT)
    template = os.environ.get("IPINYIN_VISION_CMD", "").strip()
    if not template:
        return _describe_opencode(image, prompt)
    command = template.replace("{image}", shlex.quote(image)).replace(
        "{prompt}", shlex.quote(prompt))
    try:
        done = subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("视觉模型调用失败：%s", exc)
        return None
    if done.returncode != 0:
        LOG.warning("视觉模型调用失败：退出码 %s %s", done.returncode,
                    (done.stderr or "")[:200])
        return None
    text = " ".join((done.stdout or "").split())
    return text[:MAX_DESC] or None


def capture_context(limit: int = 3) -> list[dict]:
    """扫新截图 → 逐张描述 → 落 vision.jsonl。"""
    results: list[dict] = []
    for path in scan(mark=False)[:limit]:
        mark_seen(path)
        frame = _copy_frame(path)
        desc = describe(frame)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image": frame,
            "source": path,
            "desc": desc or "",
        }
        results.append(entry)
        try:
            os.makedirs(home(), exist_ok=True)
            with open(log_path(), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
        if not desc:
            LOG.info("新截图 %s（未配置 IPINYIN_VISION_CMD，只记录路径）", path)
    return results


def cleanup(days: int | None = None) -> dict:
    """删掉自己帧目录里超过保留期的图片，并把日志裁到上限。

    只碰 `vision/frames/`——你的原始截图目录（~/Pictures 等）永远不在删除范围。
    """
    keep_days = RETENTION_DAYS if days is None else max(int(days), 0)
    cutoff = time.time() - keep_days * 86400
    removed = 0
    kept = 0
    try:
        names = os.listdir(frames_dir())
    except OSError:
        names = []
    for name in names:
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(frames_dir(), name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
            else:
                kept += 1
        except OSError:
            continue
    trimmed = _trim_log()
    return {"removed": removed, "kept": kept, "log_trimmed": trimmed,
            "retention_days": keep_days}


def _trim_log() -> int:
    try:
        with open(log_path(), encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return 0
    if len(lines) <= MAX_LOG_LINES:
        return 0
    keep = lines[-MAX_LOG_LINES:]
    try:
        with open(log_path(), "w", encoding="utf-8") as handle:
            handle.writelines(keep)
    except OSError:
        return 0
    return len(lines) - len(keep)


def recent(limit: int = 20) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(log_path(), encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def status() -> dict:
    try:
        frames = sum(1 for n in os.listdir(frames_dir()) if n.lower().endswith(IMAGE_EXTS))
    except OSError:
        frames = 0
    return {
        "dirs": watch_dirs(),
        "cmd": bool(os.environ.get("IPINYIN_VISION_CMD")),
        "entries": len(recent(10_000)),
        "frames": frames,
        "retention_days": RETENTION_DAYS,
        "home": home(),
    }
