#!/usr/bin/env bash
# 打 .deb：./packaging/build_deb.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-$(cat "$ROOT/VERSION" 2>/dev/null || echo 0.0.0)}"
MAINTAINER="${MAINTAINER:-as-workstation01 <as-workstation01@localhost>}"
PKG="ipinyin"
BUILD="$ROOT/build/deb"
STAGE="$BUILD/${PKG}_${VERSION}_all"
OUT="$ROOT/dist"

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

[ -f "$ROOT/data/lexicon.marshal" ] || { echo "先跑 tools/build_lexicon.py"; exit 1; }
[ -f "$ROOT/data/prefix.marshal" ] || { echo "先跑 tools/build_lexicon.py"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/DEBIAN" \
         "$STAGE/usr/bin" \
         "$STAGE/usr/lib/ipinyin" \
         "$STAGE/usr/share/ipinyin/data" \
         "$STAGE/usr/share/ibus/component" \
         "$STAGE/usr/share/doc/$PKG"

say "拷贝代码"
cp -r "$ROOT/src/ipinyin" "$STAGE/usr/lib/ipinyin/"
find "$STAGE/usr/lib/ipinyin" -name "__pycache__" -type d -prune -exec rm -rf {} +
install -m 755 "$ROOT/packaging/deb/ipinyin-engine" "$STAGE/usr/bin/ipinyin-engine"
install -m 755 "$ROOT/packaging/deb/ipinyin-setup" "$STAGE/usr/bin/ipinyin-setup"
install -m 755 "$ROOT/packaging/deb/ipinyin-ensure" "$STAGE/usr/bin/ipinyin-ensure"

say "拷贝词库"
cp "$ROOT/data/lexicon.tsv" "$ROOT/data/lexicon.marshal" "$ROOT/data/prefix.marshal" "$ROOT/data/s2t.marshal" \
   "$ROOT/data/user_seed.tsv" "$ROOT/data/user_bigram_seed.tsv" "$ROOT/data/yue.tsv" "$ROOT/data/yue.marshal" \
   "$STAGE/usr/share/ipinyin/data/"
install -m 644 "$ROOT/data/ipinyin.svg" "$STAGE/usr/share/ipinyin/ipinyin.svg"

say "写组件描述"
sed -e "s|@VERSION@|$VERSION|g" -e "s|@MAINTAINER@|$MAINTAINER|g" \
    "$ROOT/packaging/deb/ipinyin.xml.in" > "$STAGE/usr/share/ibus/component/ipinyin.xml"

say "文档"
cp "$ROOT/README.md" "$STAGE/usr/share/doc/$PKG/README.md"
cp "$ROOT/docs/typing-profile.md" "$STAGE/usr/share/doc/$PKG/typing-profile.md" 2>/dev/null || true
cat > "$STAGE/usr/share/doc/$PKG/copyright" <<'COPY'
IPinyin
词库来源与许可：
  pinyin-data (MIT) / phrase-pinyin-data (MIT) / jieba dict (MIT)
  CC-CEDICT (CC BY-SA 4.0, MDBG) https://www.mdbg.net/chinese/dictionary?page=cc-cedict

本程序代码：MIT。
COPY

SIZE=$(du -sk "$STAGE" | cut -f1)
sed -e "s|@VERSION@|$VERSION|g" -e "s|@MAINTAINER@|$MAINTAINER|g" -e "s|@SIZE@|$SIZE|g" \
    "$ROOT/packaging/deb/control.in" > "$STAGE/DEBIAN/control"
install -m 755 "$ROOT/packaging/deb/postinst" "$STAGE/DEBIAN/postinst"
install -m 755 "$ROOT/packaging/deb/postrm" "$STAGE/DEBIAN/postrm"

say "规范权限（避免把 umask 带进包里）"
find "$STAGE" -type d -exec chmod 755 {} +
find "$STAGE" -type f -exec chmod 644 {} +
chmod 755 "$STAGE/usr/bin/ipinyin-engine" "$STAGE/usr/bin/ipinyin-setup" "$STAGE/usr/bin/ipinyin-ensure" "$STAGE/DEBIAN/postinst" "$STAGE/DEBIAN/postrm"

say "生成 md5sums"
(cd "$STAGE" && find usr -type f -print0 | sort -z | xargs -0 md5sum > DEBIAN/md5sums)

say "打包"
mkdir -p "$OUT"
dpkg-deb --build --root-owner-group "$STAGE" "$OUT/${PKG}_${VERSION}_all.deb" >/dev/null
ls -lh "$OUT/${PKG}_${VERSION}_all.deb"
dpkg-deb --info "$OUT/${PKG}_${VERSION}_all.deb" | sed -n '1,20p'
