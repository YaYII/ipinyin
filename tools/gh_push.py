#!/usr/bin/env python3
"""在大陆网络下把本地提交推到 GitHub。

背景：
  - github.com:443 直连不通（curl 28 超时），api.github.com 可直连；
  - gh-proxy 镜像可用，但**会剥离 Authorization 头**，所以 git push 走镜像认证失败；
  - 因此改走 GitHub 的 git-data API：本地算出 blob/tree/commit 的 sha，
    通过 API 创建远端对象，再快进 ref。内容与 git 完全等价。

token 从 macOS 钥匙串读 github.com 的 internet password。

用法：
  python3 tools/gh_push.py HEAD master        # 推指定 commit
  python3 tools/gh_push.py --dry-run          # 只看会推什么
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = os.environ.get("GIT_REMOTE", "YaYII/ipinyin")


def token() -> str:
    out = subprocess.run(["security", "find-internet-password", "-s", "github.com", "-w"],
                         capture_output=True, text=True)
    tok = out.stdout.strip()
    if not tok:
        raise SystemExit("钥匙串里没找到 github.com 的凭据（个人访问令牌）")
    return tok


TOK = token()


def api(method: str, path: str, payload=None):
    url = path if path.startswith("http") else f"https://api.github.com/repos/{DEFAULT_REPO}" + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOK}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ipinyin-push",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            body = fh.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} {method} {path}\n{e.read().decode()[:600]}", file=sys.stderr)
        raise


def git(*args) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    commit_sha = git("rev-parse", args[0] if args else "HEAD")
    branch = args[1] if len(args) > 1 else "master"

    parent = git("rev-parse", f"{commit_sha}^")
    message = git("log", "-1", "--format=%B", commit_sha)
    author = {"name": git("log", "-1", "--format=%an", commit_sha),
              "email": git("log", "-1", "--format=%ae", commit_sha),
              "date": git("log", "-1", "--format=%aI", commit_sha)}

    ref = api("GET", f"/git/ref/heads/{branch}")
    remote_sha = ref["object"]["sha"]
    print(f"本地 {commit_sha[:8]}  远端 {remote_sha[:8]}  分支 {branch}")

    if remote_sha == commit_sha:
        print("远端已是最新，无需推送")
        return
    if parent != remote_sha:
        raise SystemExit(f"不能快进：本地父提交 {parent[:8]} != 远端 {remote_sha[:8]}，需先处理分歧")

    # core.quotepath=false：中文路径不转义，否则 ls-tree 按带引号的字面量查不到
    changed = git("-c", "core.quotepath=false", "diff-tree", "--no-commit-id",
                  "--name-status", "-r", commit_sha).splitlines()
    print(f"变更 {len(changed)} 个文件：{[l.split(chr(9))[-1] for l in changed]}")

    tree_items = []
    for line in changed:
        status, path = line.split("\t", 1)
        if status == "D":
            tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        mode = git("ls-tree", commit_sha, "--", path).split()[0]
        content = open(os.path.join(ROOT, path), "rb").read()
        blob = api("POST", "/git/blobs", {
            "content": base64.b64encode(content).decode(), "encoding": "base64"})
        tree_items.append({"path": path, "mode": mode, "type": "blob", "sha": blob["sha"]})

    base_tree = git("rev-parse", f"{remote_sha}^{{tree}}")
    new_tree = api("POST", "/git/trees", {"base_tree": base_tree, "tree": tree_items})
    print(f"新 tree {new_tree['sha'][:8]}")

    new_commit = api("POST", "/git/commits", {
        "message": message, "tree": new_tree["sha"], "parents": [remote_sha],
        "author": author, "committer": author})
    print(f"新 commit {new_commit['sha'][:8]}")

    if dry:
        print("（dry-run，未改 ref）")
        return

    api("PATCH", f"/git/refs/heads/{branch}", {"sha": new_commit["sha"], "force": False})
    print(f"已推送 {branch} → {new_commit['sha'][:8]}")

    # 回填本地 origin/<branch>（远端对象本地还没有，用 fetch 取回来）
    subprocess.run(["git", "-C", ROOT, "fetch",
                    f"https://YaYII:{TOK}@gh-proxy.com/https://github.com/{DEFAULT_REPO}.git",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
                   capture_output=True, text=True)
    print(f"本地 master = {git('rev-parse', '--short', 'HEAD')}  "
          f"origin/{branch} = {git('rev-parse', '--short', f'origin/{branch}')}")


if __name__ == "__main__":
    main()
