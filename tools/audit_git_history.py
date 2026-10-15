# -*- coding: utf-8 -*-
"""公开仓库前的**历史审计**：确认 git 历史里没有会话凭据 / 抓包 / 采集数据。

为什么必须做：把仓库从 private 改成 public，**暴露的是全部历史**，不只是当前工作区。
哪怕现在 .gitignore 写得很干净，只要早期某次提交塞进过 `*.har`、`snapshot/session.pkl`
或带 `Set-Cookie` 的日志，公开出去就抹不掉了（除非重写历史）。

检查三件事：
  1. 历史里**从未出现过**的敏感路径模式：`*.har`、`session.pkl`、`*_logged_in.html`、
     `analysis/**`、`7527_transport_capture/**`、`archive/exploration/{capture,analysis,har}/**`
  2. 历史所有提交的文件内容里是否有凭据模式（Cookie/会话 ID/账号密码/签名）——只扫文本类
  3. 当前**即将公开**的路径清单与体积（人工过一眼）

用法：py -3.13 tools\\audit_git_history.py [--json out.json]
退出码 1 = 发现需要处理的东西。
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "analysis", "publish_history_audit.md")
os.makedirs(os.path.dirname(REPORT), exist_ok=True)   # 公开克隆里可能没有 analysis/

FORBIDDEN_PATHS = [
    r"\.har$", r"\.chlsj$", r"\.pcap$",
    r"(^|/)session\.pkl$", r"_logged_in\.html$",
    r"(^|/)analysis/", r"(^|/)7527_transport_capture/",
    r"(^|/)archive/exploration/(capture|analysis|har|artifacts)/",
    r"(^|/)offline/server_access\.log$",
]
CONTENT_PATTERNS = [
    ("Cookie 头", r"(?i)\b(Set-Cookie|Cookie)\s*:"),
    ("会话 ID", r"(?i)ASPSESSIONID[A-Z]*="),
    ("账号密码字段", r"(?i)(UserPassword|PassWord|Pwd)\s*="),
    # 账号名用通用模式（不在公开文件里写真实账号名）：中文用户名 + 账号字段
    ("账号名", r"UserName=[^\s&\"']*[\u4e00-\u9fa5]"),
    ("签名", r"(?i)\b(sign|appId)\s*[=:]\s*[\w\-]{6,}"),
]
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".html", ".htm", ".js", ".css",
            ".asp", ".csv", ".xml", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh",
            ".bat", ".ps1", ".gitignore", ""}
MAX_BLOB = 2 * 1024 * 1024


def git(*args, binary=False):
    p = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True)
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")


def history_paths():
    """历史里出现过的所有路径（含已删除的）。"""
    out = git("log", "--all", "--pretty=format:", "--name-only", "-z")
    return {p for p in out.split("\0") if p.strip()}


def commits_touching(pattern):
    p = re.compile(pattern)
    hits = []
    for line in git("log", "--all", "--pretty=format:%h %s", "--name-only").splitlines():
        if p.search(line.strip()) and line.strip() and not line.startswith(" "):
            pass
    # 更可靠：逐提交查文件清单
    for h in git("log", "--all", "--pretty=format:%H").split():
        files = [f for f in git("show", "--pretty=format:", "--name-only", h).splitlines()
                 if f.strip()]
        for f in files:
            if p.search(f):
                hits.append((h[:8], f))
    return hits


def scan_current_tree():
    """当前工作区里**未被忽略**的文件（= 即将出现在公开仓库里的东西）。"""
    out = git("ls-files")
    return [f for f in out.splitlines() if f.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    lines = ["# 公开前的 git 历史审计", ""]
    problems = []

    # 1) 历史路径模式
    lines += ["## 1. 历史里是否出现过敏感路径", "",
              "| 模式 | 命中 |", "|---|---|"]
    for pat in FORBIDDEN_PATHS:
        hits = commits_touching(pat)
        lines.append("| `%s` | %s |" % (pat, ("**%d 处**：%s" % (len(hits), hits[:4]))
                                        if hits else "无"))
        if hits:
            problems.append(("历史路径", pat, len(hits)))

    # 2) 历史提交内容里的凭据模式（只扫文本 blob，去重）
    lines += ["", "## 2. 历史提交内容里的凭据模式", ""]
    seen_blob = set()
    content_hits = []
    blob_list = git("rev-list", "--objects", "--all").splitlines()
    for ln in blob_list:
        parts = ln.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, path = parts
        if sha in seen_blob or os.path.splitext(path)[1].lower() not in TEXT_EXT:
            continue
        seen_blob.add(sha)
        size = git("cat-file", "-s", sha).strip()
        try:
            if int(size) > MAX_BLOB:
                continue
        except ValueError:
            continue
        blob = git("cat-file", "-p", sha, binary=True)
        text = blob.decode("utf-8", "replace")
        for label, pat in CONTENT_PATTERNS:
            m = re.search(pat, text)
            if m:
                content_hits.append((path, label, m.group(0)[:40]))
                break
    if content_hits:
        lines += ["| 文件（历史版本） | 命中 |", "|---|---|"]
        for path, label, seg in content_hits[:40]:
            lines.append("| `%s` | %s |" % (path, label))
        problems.append(("历史内容", "凭据模式", len(content_hits)))
    else:
        lines.append("无命中 ✅")

    # 3) 当前即将公开的文件与体积
    tracked = scan_current_tree()
    total = sum(os.path.getsize(os.path.join(ROOT, f)) for f in tracked
                if os.path.exists(os.path.join(ROOT, f)))
    lines += ["", "## 3. 当前入库文件（即将公开）", "",
              "共 %d 个文件，%.1f MB" % (len(tracked), total / 1048576.0), "",
              "| 顶层目录 | 文件数 | 体积 |", "|---|---|---|"]
    agg = {}
    for f in tracked:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        top = f.split("/")[0] if "/" in f else "(根)"
        n, b = agg.get(top, (0, 0))
        agg[top] = (n + 1, b + os.path.getsize(p))
    for top, (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        lines.append("| `%s` | %d | %.1f MB |" % (top, n, b / 1048576.0))
    lines += ["", "## 4. 结论", ""]
    if problems:
        lines.append("**发现需要处理的东西**：%s" % problems)
        lines.append("")
        lines.append("⇒ 先处理后公开；若确要公开，只能重写历史（`git filter-repo`）或另建公开仓。")
    else:
        lines.append("历史里**没有**敏感的会话产物或抓包/采集数据；"
                     "当前入库文件也没有凭据模式命中 ✅")
        lines.append("")
        lines.append("⇒ 可以改公开。仍未公开（本就在 .gitignore）：`snapshot/`（会话与原始响应）、"
                     "`archive/exploration/{capture,analysis,har,artifacts}/`（含 Cookie 与账号名）。")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if args.json:
        io.open(args.json, "w", encoding="utf-8").write(
            json.dumps({"problems": problems, "files": len(tracked),
                        "mb": total / 1048576.0}, ensure_ascii=False, indent=1))
    print("wrote", REPORT)
    print("历史敏感路径/内容问题：%d" % len(problems))
    for p_ in problems:
        print("   ", p_)
    print("当前入库：%d 文件 / %.1f MB" % (len(tracked), total / 1048576.0))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
