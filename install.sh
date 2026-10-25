#!/usr/bin/env bash
# 安装 ipinyin 到当前用户目录（不需要 root）。
#
#   ./install.sh              安装并加入 GNOME 输入源
#   ./install.sh --no-source  只装引擎，不动输入源列表
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${IPINYIN_PREFIX:-$HOME/.local/share/ipinyin}"
UNIT="org.freedesktop.IBus.session.GNOME.service"
ADD_SOURCE=1

for arg in "$@"; do
    case "$arg" in
        --no-source) ADD_SOURCE=0 ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m错误:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 环境检查 ----------
command -v python3 >/dev/null || die "缺 python3"
python3 - <<'PY' || die "缺 python3-gi / gir1.2-ibus-1.0（sudo apt install python3-gi gir1.2-ibus-1.0）"
import gi
gi.require_version("IBus", "1.0")
from gi.repository import IBus
PY
command -v ibus >/dev/null || die "没装 ibus"

# 粤语神经模型的可选依赖：没有就退回规则+守卫，引擎照常工作
if python3 -c "import onnxruntime" 2>/dev/null; then
    NN_MODE="神经模型已就绪（int8，12MB）"
else
    NN_MODE="未启用 —— 想启用执行: pip3 install --user onnxruntime（然后 ./install.sh 重装一次）"
fi

if [ ! -f "$ROOT/data/s2t.marshal" ]; then
    say "简繁映射缺失，尝试生成"
    [ -f "$ROOT/vendor/STCharacters.txt" ] || curl -fsSL -o "$ROOT/vendor/STCharacters.txt" \
        https://raw.githubusercontent.com/BYVoid/OpenCC/master/data/dictionary/STCharacters.txt || true
    [ -f "$ROOT/vendor/STCharacters.txt" ] && python3 "$ROOT/tools/build_s2t.py" || true
fi

if [ ! -f "$ROOT/data/yue.marshal" ] && [ -f "$ROOT/vendor/yue_chars.yaml" ]; then
    say "编译粤语发音表"
    python3 "$ROOT/tools/build_yue_dict.py" || true
fi

if [ ! -f "$ROOT/data/lexicon.tsv" ]; then
    say "词库还没生成，尝试现场编译"
    "$ROOT/tools/fetch_data.sh"
    python3 "$ROOT/tools/build_lexicon.py"
fi
[ -f "$ROOT/data/lexicon.tsv" ] || die "缺少 data/lexicon.tsv"

# ---------- 落盘 ----------
say "安装到 $PREFIX"
mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/component" "$PREFIX/data"
rm -rf "$PREFIX/lib/ipinyin"
cp -r "$ROOT/src/ipinyin" "$PREFIX/lib/"
cp "$ROOT/data/lexicon.tsv" "$PREFIX/data/lexicon.tsv"
# 紧凑快照与前缀表：引擎启动读它们，比解析文本快 3~4 倍
cp "$ROOT"/data/*.marshal "$PREFIX/data/" 2>/dev/null || true
# 个人词频：seed 是你历史输入里挖出来的词，user_model.tsv 是运行时边用边学的
cp "$ROOT/data/user_seed.tsv" "$PREFIX/data/user_seed.tsv" 2>/dev/null || true
cp "$ROOT/data/user_bigram_seed.tsv" "$PREFIX/data/user_bigram_seed.tsv" 2>/dev/null || true
touch "$PREFIX/data/user_model.tsv"
cp "$ROOT/data/ipinyin.svg" "$PREFIX/data/ipinyin.svg"
cp "$ROOT/data/s2t.marshal" "$PREFIX/data/s2t.marshal"
cp "$ROOT/data/yue.tsv" "$PREFIX/data/yue.tsv"
cp "$ROOT/data/yue.marshal" "$PREFIX/data/yue.marshal"
[ -f "$ROOT/data/yue_phrase.marshal" ] && cp "$ROOT/data/yue_phrase.marshal" "$PREFIX/data/yue_phrase.marshal"
# 粤语神经模型（int8 量化版，随仓库分发；没有它自动退回规则+守卫）
[ -f "$ROOT/data/yue_nn.onnx" ] && cp "$ROOT/data/yue_nn.onnx" "$PREFIX/data/yue_nn.onnx"
[ -f "$ROOT/data/yue_nn_vocab.json" ] && cp "$ROOT/data/yue_nn_vocab.json" "$PREFIX/data/yue_nn_vocab.json"
install -m 755 "$ROOT/bin/ipinyin-engine" "$PREFIX/bin/ipinyin-engine"
install -m 755 "$ROOT/bin/ipinyin-ensure" "$PREFIX/bin/ipinyin-ensure"
sed "s|@PREFIX@|$PREFIX|g" "$ROOT/packaging/ipinyin.xml.in" > "$PREFIX/component/ipinyin.xml"

# ---------- 让 ibus-daemon 认识这个组件 ----------
# IBus 默认只扫 /usr/share/ibus/component，用户目录不被扫描，
# 所以用 IBUS_COMPONENT_PATH 追加（必须保留系统目录，否则别的输入法会消失）。
COMPONENT_PATH="/usr/share/ibus/component:$PREFIX/component"
DROPDIR="$HOME/.config/systemd/user/$UNIT.d"
mkdir -p "$DROPDIR"
cat > "$DROPDIR/ipinyin.conf" <<EOF
[Service]
Environment="IBUS_COMPONENT_PATH=$COMPONENT_PATH"
EOF

rm -f "$HOME/.cache/ibus/bus/registry"

restart_ibus() {
    # 手工起过的 ibus-daemon 不会被 systemd 接管，留着会让 unit 启动失败
    # （"current session already has an ibus-daemon"），先清掉。
    if systemctl --user cat "$UNIT" >/dev/null 2>&1; then
        local main_pid
        main_pid=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null || echo 0)
        for pid in $(pgrep -x ibus-daemon); do
            [ "$pid" = "$main_pid" ] || kill "$pid" 2>/dev/null || true
        done
        systemctl --user daemon-reload
        # 组件缓存会记住旧的组件列表，不删就看不到新引擎
        rm -f "$HOME/.cache/ibus/bus/registry"
        for _ in 1 2 3; do
            systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
            systemctl --user restart "$UNIT" >/dev/null 2>&1 || true
            sleep 2
            ibus list-engine >/dev/null 2>&1 && return 0
            systemctl --user start "$UNIT" >/dev/null 2>&1 || true
            sleep 1
        done
        return 1
    fi
    say "没找到 systemd 单元，直接重启 ibus-daemon"
    rm -f "$HOME/.cache/ibus/bus/registry"
    IBUS_COMPONENT_PATH="$COMPONENT_PATH" ibus-daemon -drx || true
    sleep 2
}

say "重启 IBus 守护进程"
restart_ibus || true

# ---------- 常驻服务：引擎崩了 2 秒自动拉起 ----------
sed "s|@PREFIX@|$PREFIX|g" "$ROOT/packaging/ipinyin-engine.service" > "$HOME/.config/systemd/user/ipinyin-engine.service"
systemctl --user daemon-reload
systemctl --user enable ipinyin-engine.service >/dev/null 2>&1 || true
systemctl --user restart ipinyin-engine.service >/dev/null 2>&1 || true

# ---------- 预加载，避开首次切换的 spawn 竞态 ----------
if command -v gsettings >/dev/null && gsettings get org.freedesktop.ibus.general preload-engines >/dev/null 2>&1; then
    say "加入 IBus 预加载列表"
    python3 "$ROOT/tools/ibus_preload.py" add || true
fi

# ---------- 加入 GNOME 输入源 ----------
if [ "$ADD_SOURCE" = 1 ] && command -v gsettings >/dev/null; then
    if gsettings get org.gnome.desktop.input-sources sources >/dev/null 2>&1; then
        say "登记到 GNOME 输入源"
        python3 "$ROOT/tools/gnome_source.py" add || true
    fi
fi

# ---------- 自检 ----------
if ibus list-engine 2>/dev/null | grep -q "ipinyin"; then
    say "安装完成：ibus 已识别 ipinyin"
    ibus list-engine | grep ipinyin
else
    die "ibus list-engine 里没有 ipinyin，检查 $HOME/.cache/ipinyin/engine.log"
fi

if [ -f "$PREFIX/data/yue_nn.onnx" ]; then
    say "粤语转换: $NN_MODE"
else
    say "粤语转换: 神经模型文件缺失（仓库 data/yue_nn.onnx）"
fi

cat <<'EOF'

下一步：
  * 按 Super+Space 切到「IPinyin 拼音」，或直接执行： ibus engine ipinyin
  * 若切换列表里没看到它，注销重登一次（GNOME 只启动时读输入源列表）
  * 卸载：./uninstall.sh
EOF
