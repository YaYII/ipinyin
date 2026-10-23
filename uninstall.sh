#!/usr/bin/env bash
# 卸载 ipinyin：删文件、撤注册、恢复 IBus 组件搜索路径。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${IPINYIN_PREFIX:-$HOME/.local/share/ipinyin}"
UNIT="org.freedesktop.IBus.session.GNOME.service"

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

systemctl --user disable --now ipinyin-engine.service >/dev/null 2>&1 || true
rm -f "$HOME/.config/systemd/user/ipinyin-engine.service"

if command -v gsettings >/dev/null 2>&1; then
    python3 "$ROOT/tools/gnome_source.py" remove || true
    python3 "$ROOT/tools/ibus_preload.py" remove || true
fi

rm -f "$HOME/.config/systemd/user/$UNIT.d/ipinyin.conf"
rmdir "$HOME/.config/systemd/user/$UNIT.d" 2>/dev/null || true
say "删除 $PREFIX"
rm -rf "$PREFIX"
rm -f "$HOME/.cache/ibus/bus/registry"

rm -f "$HOME/.cache/ibus/bus/registry"
if systemctl --user cat "$UNIT" >/dev/null 2>&1; then
    systemctl --user daemon-reload
    systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
    systemctl --user restart "$UNIT" >/dev/null 2>&1 || systemctl --user start "$UNIT" >/dev/null 2>&1 || true
else
    ibus-daemon -drx || true
fi
say "已卸载（若当前正用着 ipinyin，先 Super+Space 切回别的输入法）"
