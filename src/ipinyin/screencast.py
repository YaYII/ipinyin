"""ScreenCast 常驻截屏：授权一次（persist_mode=2），之后靠 restore_token 静默恢复。

多显示器：授权时选多个屏幕，Start 会返回多条串流，每条带 position/size。
引擎通过 `do_set_cursor_location` 把输入游标坐标写到 cursor.json，
这里按「游标落在哪条串流的矩形里」决定抓哪一屏——你在哪个屏幕打字就抓哪个。

取帧走 portal 的 OpenPipeWireRemote 拿 fd，再用 GStreamer 的 pipewiresrc fd= 读一帧。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time

LOG = logging.getLogger("ipinyin")

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST = "org.freedesktop.portal.ScreenCast"


def home() -> str:
    from . import vision

    return vision.home()


def state_path() -> str:
    return os.path.join(home(), "screencast.json")


def cursor_path() -> str:
    return os.path.join(home(), "cursor.json")


def incoming_dir() -> str:
    return os.path.join(home(), "incoming")


def _load_state() -> dict:
    try:
        with open(state_path(), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _save_state(**fields) -> None:
    data = _load_state()
    data.update(fields)
    data["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        os.makedirs(home(), exist_ok=True)
        with open(state_path(), "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
    except OSError:
        pass


def write_cursor(x: int, y: int, width: int = 0, height: int = 0) -> None:
    """引擎在每次输入框光标变化时调用（屏幕坐标）。"""
    try:
        os.makedirs(home(), exist_ok=True)
        with open(cursor_path(), "w", encoding="utf-8") as handle:
            json.dump({"x": int(x), "y": int(y), "w": int(width), "h": int(height),
                       "ts": time.time()}, handle)
    except OSError:
        pass


def read_cursor() -> tuple[int, int] | None:
    try:
        with open(cursor_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data["x"]), int(data["y"])
    except (OSError, ValueError, KeyError):
        return None


def authorize(multiple: bool = True, timeout: float = 180.0) -> dict:
    """跑一次授权，返回 {session, streams, token}；失败返回 {}。"""
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    loop = GLib.MainLoop()
    state: dict = {}

    def on_response(conn, sender, path, iface, signal, params, user_data=None):
        state["code"], state["results"] = params.unpack()
        loop.quit()

    def call(method: str, args) -> dict:
        sub = bus.signal_subscribe(
            PORTAL, "org.freedesktop.portal.Request", "Response", None, None,
            Gio.DBusSignalFlags.NONE, on_response, None)
        bus.call_sync(PORTAL, PORTAL_PATH, SCREENCAST, method, args, None,
                      Gio.DBusCallFlags.NONE, -1, None)
        GLib.timeout_add(int(timeout * 1000), loop.quit)
        loop.run()
        bus.signal_unsubscribe(sub)
        return state.pop("results", {}) or {}

    opts = {"handle_token": GLib.Variant("s", "ipv2"),
            "session_handle_token": GLib.Variant("s", "ips2")}
    res = call("CreateSession", GLib.Variant("(a{sv})", (opts,)))
    session = res.get("session_handle")
    if not session:
        return {}

    token = _load_state().get("restore_token")
    sel = {"types": GLib.Variant("u", 1),          # 1 = 显示器
           "multiple": GLib.Variant("b", multiple),
           "cursor_mode": GLib.Variant("u", 2),
           "persist_mode": GLib.Variant("u", 2)}   # 持久授权：必须有，否则拿不到 token
    if token:
        sel["restore_token"] = GLib.Variant("s", token)
    call("SelectSources", GLib.Variant("(oa{sv})", (session, sel)))
    res = call("Start", GLib.Variant("(osa{sv})", (session, "", {})))
    streams = []
    for node, props in (res.get("streams") or []):
        pos = props.get("position") or (0, 0)
        size = props.get("size") or (0, 0)
        streams.append({"node": int(node), "x": int(pos[0]), "y": int(pos[1]),
                        "w": int(size[0]), "h": int(size[1])})
    token = res.get("restore_token") or token
    _save_state(restore_token=token, streams=streams, session=session)
    return {"session": session, "streams": streams, "token": token, "bus": bus}


def pick_stream(streams: list[dict]) -> dict | None:
    """按输入游标位置选屏幕；没有游标信息就选第一个。"""
    if not streams:
        return None
    cursor = read_cursor()
    if cursor is None:
        return streams[0]
    x, y = cursor
    for stream in streams:
        if (stream["x"] <= x < stream["x"] + max(stream["w"], 1)
                and stream["y"] <= y < stream["y"] + max(stream["h"], 1)):
            return stream
    return streams[0]


def _open_fd(bus, session: str) -> int | None:
    """OpenPipeWireRemote：D-Bus 的 h 类型是句柄索引，要配 UnixFDList 取真 fd。"""
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    try:
        result, fd_list = bus.call_with_unix_fd_list_sync(
            PORTAL, PORTAL_PATH, SCREENCAST, "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (session, {})), None,
            Gio.DBusCallFlags.NONE, -1, None, None)
        index = result.unpack()[0]
        return fd_list.get(index)
    except Exception as exc:  # noqa: BLE001 - 任何失败都只记日志
        LOG.warning("OpenPipeWireRemote 失败：%s", exc)
        return None


def grab(fd: int, node: int, target: str, timeout: int = 15) -> bool:
    """用 pipewiresrc 抓一帧存成 PNG。"""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    if not Gst.is_initialized():
        Gst.init(None)
    pipeline = Gst.parse_launch(
        f"pipewiresrc fd={fd} path={node} num-buffers=1 ! videoconvert ! "
        f"pngenc ! filesink location={target}")
    pipeline.set_state(Gst.State.PLAYING)
    msg = pipeline.get_bus().timed_pop_filtered(
        timeout * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    ok = bool(msg and msg.type == Gst.MessageType.EOS)
    if not ok and msg is not None:
        LOG.warning("抓帧失败：%s", msg.parse_error()[1])
    pipeline.set_state(Gst.State.NULL)
    return ok and os.path.exists(target) and os.path.getsize(target) > 0


def run(interval: float = 20.0) -> int:
    """常驻：授权 → 按游标选屏 → 定期抓帧到 incoming/。"""
    info = authorize()
    if not info.get("streams"):
        print("授权失败：没有拿到屏幕串流")
        return 1
    streams = info["streams"]
    fd = _open_fd(info["bus"], info["session"])
    if fd is None:
        print("拿不到 PipeWire fd，抓帧不可用")
        return 1
    os.makedirs(incoming_dir(), exist_ok=True)
    print(f"已授权：{len(streams)} 块屏幕，token {'有' if info.get('token') else '无'}")
    for stream in streams:
        print(f"  节点 {stream['node']}: {stream['w']}x{stream['h']} @ ({stream['x']},{stream['y']})")
    print(f"每 {interval:.0f} 秒抓一帧到 {incoming_dir()}（Ctrl+C 退出）")

    stop = {"flag": False}

    def _stop(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop["flag"]:
        stream = pick_stream(streams)
        if stream is None:
            break
        target = os.path.join(incoming_dir(), f"frame-{int(time.time() * 1000)}.png")
        if grab(fd, stream["node"], target):
            LOG.info("抓帧 %s（节点 %s）", os.path.basename(target), stream["node"])
        time.sleep(interval)
    return 0


def status() -> dict:
    data = _load_state()
    try:
        frames = len([n for n in os.listdir(incoming_dir()) if n.endswith(".png")])
    except OSError:
        frames = 0
    return {"authorized": bool(data.get("restore_token")),
            "streams": data.get("streams", []),
            "frames": frames,
            "cursor": read_cursor()}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run(float(os.environ.get("IPINYIN_VISION_INTERVAL", "20")))


if __name__ == "__main__":
    raise SystemExit(main([]))
