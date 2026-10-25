#!/bin/bash
# 构建 macOS 输入法：Python 引擎打包进 Resources/engine，Swift IMKit 外壳编译进
# IPinyin.app，装机到 ~/Library/Input Methods 并注册。全程 /usr/bin/python3，无第三方依赖。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_NAME="IPinyin"
APP_ID="org.yayii.ipinyin"
BUILD="$ROOT/build/macos"
APP="$BUILD/$APP_NAME.app"
INSTALL_DIR="$HOME/Library/Input Methods"
INSTALLED="$INSTALL_DIR/$APP_NAME.app"

echo "==> 打包引擎（src + tools + data）"
rm -rf "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine"
RES="$APP/Contents/Resources/engine"
rsync -a --exclude '__pycache__' --exclude '*.eval.marshal' \
    "$ROOT/src" "$ROOT/tools" "$ROOT/data" "$RES/"

echo "==> 编译 IMKit 外壳"
cp "$ROOT/tools/macos/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/tools/macos/ipinyin.pdf" "$APP/Contents/Resources/ipinyin.pdf"
swiftc -O -o "$APP/Contents/MacOS/$APP_NAME" \
    "$ROOT/tools/macos/main.swift" "$ROOT/tools/macos/Controller.swift" "$ROOT/tools/macos/Engine.swift" \
    -framework Cocoa -framework Carbon -framework InputMethodKit

echo "==> 签名（ad-hoc）"
codesign --force --sign - "$APP" 2>/dev/null

echo "==> 装机 $INSTALLED"
mkdir -p "$INSTALL_DIR"
[ -d "$INSTALLED" ] && killall "$APP_NAME" 2>/dev/null || true
rm -rf "$INSTALLED"
cp -R "$APP" "$INSTALLED"

echo "==> 注册输入源"
"$INSTALLED/Contents/MacOS/$APP_NAME" --register --select || true

echo
echo "完成。若系统设置里还没出现「IPinyin 拼音」："
echo "  1. 注销重登（首次注册最稳）"
echo "  2. 系统设置 → 键盘 → 输入法 → 编辑 → 添加「中文，简体 → IPinyin 拼音」"
echo "  3. 控制中心输入法菜单里切到 IPinyin，任意输入框直接打拼音"
