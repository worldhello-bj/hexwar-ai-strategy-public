# -*- coding: utf-8 -*-
"""分段长跑驱动：一份配置 → 若干段，每段落检查点/报告并带一次重试。

为什么需要它：
- 本仓库**只在进程结束时写检查点**，而一次跑几小时 ⇒ 中途被打断就全丢（已踩过：
  终端被清理导致 96 局白跑）。
- 所以把长跑切成段：每段独立进程、落一次 `ckpt_seg_NN.json` / `report_seg_NN.json`，
  下一段 `resume` 上一段；某段失败重试一次，仍失败就停下（已完成的段都留着）。
- ★ 段配置全部生成在**系统临时目录**里，不会污染 `configs/`（以前那里堆了 16 个 `_seg_*.json`）。

用法：
    py -3.13 run_long.py --config configs/ppo_match.json --segments 6 --episodes 50
    py -3.13 run_long.py --config configs/ppo_match.json --segments 6 --episodes 50 --dry-run
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_segments(base, segments, episodes, out_dir):
    """按 `base` 生成 `segments` 段配置（写进 `out_dir`），返回它们的路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, previous = [], None
    stem = Path(base["checkpoint"]).stem.replace(".json", "") or "run"
    for index in range(1, segments + 1):
        cfg = dict(base)
        cfg["episodes"] = episodes
        cfg["note"] = ("%s 第 %d/%d 段（每段 %d 局）。段小是为了最坏只丢一段；"
                       "每段落检查点与报告，进度台账跨段续写。" %
                       (base.get("note", stem)[:60], index, segments, episodes))
        cfg["checkpoint"] = "%s_seg_%02d.json" % (base["checkpoint"][:-5], index)
        cfg["report"] = "%s_seg_%02d.json" % (base["report"][:-5], index)
        if previous:
            cfg["resume"] = previous
        previous = cfg["checkpoint"]
        path = out_dir / ("seg_%02d.json" % index)
        io.open(path, "w", encoding="utf-8").write(
            json.dumps(cfg, ensure_ascii=False, indent=1) + "\n")
        paths.append(path)
    return paths


def run_segment(path, log_path):
    started = time.time()
    with io.open(log_path, "wb") as stream:
        code = subprocess.call([sys.executable, "run_train.py", "train", "--algo", "ppo",
                                "--config", str(path)],
                               cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT)
    return code, time.time() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="基础配置（configs/xxx.json）")
    parser.add_argument("--segments", type=int, default=6, help="分几段")
    parser.add_argument("--episodes", type=int, default=50, help="每段多少局（建议偶数）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    parser.add_argument("--retries", type=int, default=1, help="单段失败重试次数")
    args = parser.parse_args(argv)

    base = json.loads(args.config.read_text("utf-8"))
    out_dir = Path(tempfile.mkdtemp(prefix="hexwar_seg_"))
    paths = build_segments(base, args.segments, args.episodes, out_dir)
    stamp = Path(base.get("progress", "analysis/long/progress.jsonl")).parent / "segments.txt"
    if not args.dry_run:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        # 检查点/报告父目录也得先建：训练器用 "x" 模式写文件，目录不在会直接报错退出
        for key in ("checkpoint", "report", "progress"):
            if base.get(key):
                Path(base[key]).parent.mkdir(parents=True, exist_ok=True)

    def note(text):
        line = "%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
        with io.open(stamp, "a", encoding="utf-8") as stream:
            stream.write(line)
        print(line.rstrip())

    note("start: %d 段 × %d 局，基础配置 %s" % (args.segments, args.episodes, args.config))
    note("段配置目录：%s" % out_dir)
    if args.dry_run:
        for path in paths:
            print("py -3.13 run_train.py train --algo ppo --config %s" % path)
        return 0

    for path in paths:
        for attempt in range(args.retries + 1):
            code, seconds = run_segment(path, out_dir / (path.stem + ".log"))
            note("%s attempt=%d exit=%d 用时 %.1f min" % (path.stem, attempt + 1, code,
                                                          seconds / 60.0))
            if code == 0:
                break
            if attempt < args.retries:
                time.sleep(30)
        else:
            note("STOP：%s 连续失败，已完成段与检查点保留" % path.stem)
            return 1
    note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
