"""扫密钥：默认扫工作区里的已跟踪文件，--staged 扫暂存区。

AI 通道的 key 只该存在于 ~/.dsh/.credentials.yaml 或环境变量里，
绝不进仓库（用户明确要求过）。给 pre-commit 与 `make secrets` 用。
"""
from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "API key (sk-)"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"AKIA[0-9A-Z]{12,}"), "AWS access key"),
    (re.compile(r"xox[bp]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "私钥"),
    (re.compile(r"OPENCODE_GO_API_KEY\s*[:=]\s*[\"']?[A-Za-z0-9_-]{16,}"), "opencode key"),
)
# 测试里的假 key / 文档里的占位符
ALLOW = ("sk-test-", "example", "xxxx", "your-", "<", "…")


def hits_in(text: str, name: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for pattern, label in PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if any(marker in value.lower() for marker in ALLOW):
                continue
            found.append((name, label, value[:10] + "…"))
    return found


def staged_text() -> str:
    done = subprocess.run(["git", "diff", "--cached", "-U0"], capture_output=True, text=True)
    return done.stdout


def tracked_files() -> list[str]:
    done = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    return [line for line in done.stdout.splitlines() if line]


def main(argv: list[str]) -> int:
    problems: list[tuple[str, str, str]] = []
    if "--staged" in argv:
        problems = hits_in(staged_text(), "暂存区")
    else:
        for name in tracked_files():
            try:
                with open(name, encoding="utf-8", errors="ignore") as handle:
                    problems += hits_in(handle.read(), name)
            except OSError:
                continue
    if problems:
        print("发现疑似密钥（不该提交）：", file=sys.stderr)
        for name, label, preview in problems:
            print(f"  {name}: {label} {preview}", file=sys.stderr)
        return 1
    print("没有发现密钥")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
