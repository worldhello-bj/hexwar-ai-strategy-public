# -*- coding: utf-8 -*-
"""跟读训练进度台账（`env.neural_train` 的 `progress_path`），打印速率/ETA/近期战况。

用法：
    py -3.13 tools/monitor_train.py --progress train/analysis/longrun/progress.jsonl --total 1000
    py -3.13 tools/monitor_train.py --progress ... --total 1000 --follow --interval 15

设计要点：
- 台账是**追加写**的 JSONL，每完成一局一行；本工具只读、不碰训练进程。
- 支持**分段落跑**：台账跨进程续写，`attempted` 单调递增即可（分段重启不会重头数）。
- `--follow` 以固定间隔重打一行摘要（不刷屏、重定向到文件也好看）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SIDES = ("BLUE", "RED")


def read_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def summarize(rows, total, window=50, now=None, gap_limit=300.0):
    """把台账整理成一块可直接读的报告（纯函数，便于测试）。

    ★ 速率/ETA 只放**近期窗口**（最后 `window` 行）：长跑中间可能因故停了几个小时
    （终端被清理、机器休眠），用“从第一行算起”的累计速率会把 ETA 拉到离谱的级别。
    """
    now = time.time() if now is None else now
    if not rows:
        return {"done": 0, "total": total, "percent": 0.0, "rate_per_min": None,
                "eta_minutes": None, "updates": 0, "recent": None, "last_wall": None,
                "stalled_seconds": None, "overall_rate_per_min": None}
    done = rows[-1].get("attempted") or len(rows)
    first_wall = rows[0].get("wall")
    last_wall = rows[-1].get("wall")
    overall = None
    if (first_wall is not None and last_wall is not None
            and last_wall > first_wall and done > 1):
        overall = (done - 1) / ((last_wall - first_wall) / 60.0)
    # ★ 只看**尾部连续段**：相邻两行相隔超过 `gap_limit` 秒就当成中断
    #   （终端被清理、机器休眠、手动暂停……）。否则一次 4.7 h 的停机能把 ETA 从
    #   10 h 拉成 84 h（已踩过）。
    tail_rows = [rows[-1]]
    for row in reversed(rows[:-1]):
        if tail_rows[-1].get("wall", 0) - row.get("wall", 0) > gap_limit:
            break
        tail_rows.append(row)
        if len(tail_rows) >= window:
            break
    tail_rows.reverse()
    rate = None
    if len(tail_rows) >= 2:
        span = tail_rows[-1].get("wall", 0) - tail_rows[0].get("wall", 0)
        if span > 0:
            rate = (len(tail_rows) - 1) / (span / 60.0)
    eta = None
    if rate and total:
        eta = max(0.0, (total - done) / rate)
    tail = [r for r in tail_rows if r.get("R") is not None]
    recent = None
    if tail:
        recent = {
            "n": len(tail),
            "mean_R": sum(r["R"] for r in tail) / len(tail),
            "mean_own": sum(r["own"] for r in tail) / len(tail),
            "wins": sum(1 for r in tail if r.get("result") == "win"),
            "losses": sum(1 for r in tail if r.get("result") == "loss"),
            "draws": sum(1 for r in tail if r.get("result") == "draw"),
            "mean_episode_seconds": sum((r.get("episode_seconds") or 0) for r in tail) / len(tail),
            "decisions": tail[-1].get("decisions"),
        }
    return {"done": done, "total": total,
            "percent": (100.0 * done / total) if total else None,
            "rate_per_min": rate, "overall_rate_per_min": overall,
            "eta_minutes": eta,
            "updates": rows[-1].get("updates"), "recent": recent,
            "last_wall": last_wall,
            "stalled_seconds": (now - last_wall) if last_wall else None}


def format_line(report):
    if not report["done"]:
        return "还没有进度（台账为空）：训练进程可能仍在规则门/第一批 rollout 里"
    parts = ["%d/%d 局 (%.1f%%)" % (report["done"], report["total"], report["percent"] or 0.0)]
    if report["rate_per_min"]:
        parts.append("近期 %.2f 局/min" % report["rate_per_min"])
    if report.get("overall_rate_per_min"):
        parts.append("累计 %.2f" % report["overall_rate_per_min"])
    if report["eta_minutes"] is not None:
        parts.append("ETA %.1f h" % (report["eta_minutes"] / 60.0))
    parts.append("更新 %s 次" % report["updates"])
    recent = report["recent"]
    if recent:
        parts.append("近 %d 局 R=%.2f 分差胜/平/负=%d/%d/%d 单局 %.0f s"
                     % (recent["n"], recent["mean_R"], recent["wins"], recent["draws"],
                        recent["losses"], recent["mean_episode_seconds"]))
    if report["stalled_seconds"] is not None:
        parts.append("台账距今 %.0f s" % report["stalled_seconds"])
    return " | ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--progress", required=True, type=Path, help="进度台账（JSONL）")
    parser.add_argument("--total", type=int, default=None, help="本次计划总局数（算 ETA 用）")
    parser.add_argument("--window", type=int, default=50, help="近期统计窗口（局）")
    parser.add_argument("--follow", action="store_true", help="持续跟读")
    parser.add_argument("--interval", type=float, default=15.0, help="跟读间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只打一行摘要（适合被别的脚本调用）")
    args = parser.parse_args(argv)

    while True:
        rows = read_rows(args.progress)
        total = args.total or (int(rows[-1]["attempted"]) if rows else 0)
        report = summarize(rows, total, window=args.window)
        print(format_line(report))
        sys.stdout.flush()
        if not args.follow or args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
