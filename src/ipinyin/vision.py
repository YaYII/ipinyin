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


def home() -> str:
    return os.path.join(config.data_home(), "vision")


def state_path() -> str:
    return os.path.join(home(), "state.json")


def log_path() -> str:
    return os.path.join(home(), "vision.jsonl")


def watch_dirs() -> list[str]:
    """默认看 GNOME 截图目录与项目自己的收件箱。"""
    override = os.environ.get("IPINYIN_VISION_DIRS")
    if override:   # 显式指定就只用指定的（测试与自建目录都用这条）
        return [d for d in override.split(os.pathsep) if d and os.path.isdir(d)]
    dirs = [
        os.path.expanduser("~/Pictures/截图"),
        os.path.expanduser("~/Pictures/Screenshots"),
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


def scan() -> list[str]:
    """返回上次之后新增的截图（按修改时间排序）。"""
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
    state["seen"] = seen
    _save_state(state)
    return [path for _, path in fresh]


def describe(image: str) -> str | None:
    """调用视觉模型命令；没配置或失败返回 None（绝不抛）。"""
    template = os.environ.get("IPINYIN_VISION_CMD", "").strip()
    if not template:
        return None
    prompt = os.environ.get("IPINYIN_VISION_PROMPT", DEFAULT_PROMPT)
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
    for path in scan()[:limit]:
        desc = describe(path)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image": path,
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
    return {
        "dirs": watch_dirs(),
        "cmd": bool(os.environ.get("IPINYIN_VISION_CMD")),
        "entries": len(recent(10_000)),
        "home": home(),
    }
