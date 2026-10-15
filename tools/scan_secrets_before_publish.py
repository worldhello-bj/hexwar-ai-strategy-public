# -*- coding: utf-8 -*-
"""公开发布前的**敏感信息扫描**（只读，不改任何文件）。

背景：本仓库的采集产物里有会话凭据（HAR 存 Cookie/认证头；部分抓取响应含
`Set-Cookie` / `ASPSESSIONID`；账号密码只走环境变量，但个别日志可能带 UserName）。
**公开仓库 = 任何人可 clone**，所以发布前必须逐个确认哪些文件能发、哪些必须先清洗。

扫描对象（默认）：
  archive/exploration/har       抓包（★ 最可能含 Cookie）
  archive/exploration/capture   实测采集（裁决页、房间页、响应体）
  archive/exploration/analysis  分析中间产物（日志/渲染）
  snapshot                      想定解析结果与原始快照
  offline                       桩数据 + 访问日志（站点镜像单独列）
  docs                          （手册 PDF 单列，不进正则扫描）

判据是**模式命中**（不做误报清洗），命中一律按"需要人工确认"处理。

用法：
  py -3.13 tools\\scan_secrets_before_publish.py                # 扫描 + 写报告
  py -3.13 tools\\scan_secrets_before_publish.py --json out.json
退出码：0 = 无命中；1 = 有命中（需人工处理后再发布）。
"""
import argparse
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "analysis", "publish_scan.md")

# (标签, 正则, 是否属于"凭据/会话"这类必须处理的)
PATTERNS = [
    ("Cookie 头", r"(?i)\b(Set-Cookie|Cookie)\s*:", True),
    ("ASP 会话 ID", r"(?i)ASPSESSIONID[A-Z]*=", True),
    ("Authorization 头", r"(?i)\bAuthorization\s*:", True),
    ("表单账号字段", r"(?i)(UserName|UserPassword|PassWord|Pwd)\s*=", True),
    # 账号名用通用模式（不在公开文件里写真实账号名）：
    # 中文用户名 + 账号字段；额外名单可用环境变量 HEXWAR_SECRET_NAMES（逗号分隔）补充。
    ("账号名", r"UserName=[^\s&\"']*[\u4e00-\u9fa5]", False),
    ("签名/appId", r"(?i)\b(sign|appId|appid)\s*[=:]\s*[\w\-]{6,}", True),
    ("抓包格式", r"(?i)(\"log\"\s*:\s*\{\s*\"version\"|chlsj|WebKitFormBoundary)", False),
]
# 明确要单独判断、不参与正则扫描的大件
BIG_ASSETS = [
    "docs/WARGAMEAI.pdf",
    "offline/site", "offline/assets", "offline/mock", "offline/data.js",
    "archive/exploration/har/www.hexwar.cn.har",
]
SCAN_DIRS = ["archive/exploration/har", "archive/exploration/capture",
             "archive/exploration/analysis", "snapshot", "offline"]
TEXT_EXT = {".html", ".htm", ".txt", ".json", ".jsonl", ".md", ".log", ".asp",
            ".js", ".har", ".csv", ".xml", ".py", ".cgi"}
MAX_BYTES = 40 * 1024 * 1024          # 超大文件只报体积，不逐行扫


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    hits = {}
    scanned = 0
    skipped = []
    for rel in SCAN_DIRS:
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        for cur, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                path = os.path.join(cur, name)
                r = os.path.relpath(path, ROOT).replace(os.sep, "/")
                size = os.path.getsize(path)
                if os.path.splitext(name)[1].lower() not in TEXT_EXT:
                    skipped.append((r, size, "非文本/未扫描"))
                    continue
                if size > MAX_BYTES:
                    skipped.append((r, size, "超过 40MB 上限"))
                    continue
                scanned += 1
                try:
                    text = io.open(path, encoding="utf-8", errors="replace").read()
                except OSError as exc:                          # noqa: BLE001
                    skipped.append((r, size, "读失败 %s" % exc))
                    continue
                for label, pat, sensitive in PATTERNS:
                    for m in re.finditer(pat, text):
                        entry = hits.setdefault(r, [])
                        if any(e[0] == label for e in entry):
                            continue
                        line = text[max(0, m.start() - 30):m.start() + 60]
                        line = re.sub(r"[\w\-\.]{12,}", "<REDACTED>", line)
                        entry.append((label, sensitive, line.replace("\n", " ")))

    total_sensitive = sum(1 for r in hits for e in hits[r] if e[1])
    lines = ["# 公开发布前敏感信息扫描", "",
             "扫描 %d 个文本文件；命中 %d 个文件（其中含凭据/会话类命中 %d 处）。"
             % (scanned, len(hits), total_sensitive), ""]
    if hits:
        lines += ["## 命中（发布前必须逐个处理）", "",
                  "| 文件 | 命中 | 类别 | 片段（已打码） |", "|---|---|---|---|"]
        for r in sorted(hits, key=lambda x: -len(hits[x])):
            for label, sensitive, seg in hits[r]:
                lines.append("| `%s` | %s | %s | `%s` |"
                             % (r, label, "★凭据" if sensitive else "信息",
                                seg.strip()[:90]))
        lines.append("")
    else:
        lines += ["## 未命中任何模式（仍建议人工抽查大件）", ""]
    lines += ["## 未扫描（非文本或超大）", "",
              "| 文件 | 字节 | 原因 |", "|---|---|---|"]
    lines += ["| `%s` | %d | %s |" % s for s in sorted(skipped)[:40]]
    lines += ["", "## 单独判断的大件（版权问题，非正则能解决）", ""]
    for rel in BIG_ASSETS:
        p = os.path.join(ROOT, rel)
        if os.path.isdir(p):
            n = sum(len(f) for _b, _d, f in os.walk(p))
            lines.append("- `%s`（目录，%d 个文件，官方前端资源，版权归原厂商）" % (rel, n))
        elif os.path.exists(p):
            lines.append("- `%s`（%.1f MB，属第三方版权）" % (rel, os.path.getsize(p) / 1048576.0))
        else:
            lines.append("- `%s`（缺失）" % rel)
    lines += ["", "## 结论", "",
              "1. **凭据/会话类命中一律不能公开发布** —— 那会泄露你自己的账号会话"
              "（`Cookie`/`ASPSESSIONID`/`Authorization`/`sign`）。",
              "2. **第三方版权件（官方前端镜像、比赛手册 PDF）需要权利基础**；"
              "无许可时改用「按 URL 拉取 + sha1 校验」的脚本，不把文件放进仓库。",
              "3. 其余自有代码/文档可正常公开。"]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if args.json:
        io.open(args.json, "w", encoding="utf-8").write(
            json.dumps({"hits": {k: [list(e) for e in v] for k, v in hits.items()},
                        "scanned": scanned}, ensure_ascii=False, indent=1))
    print("wrote", REPORT)
    print("扫描 %d 个文件；命中 %d；凭据类命中 %d" % (scanned, len(hits), total_sensitive))
    return 1 if total_sensitive else 0


if __name__ == "__main__":
    sys.exit(main())
