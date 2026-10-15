# -*- coding: utf-8 -*-
"""公开克隆自检：**临时隐藏本地完整采集**，只用随仓库的 `scenario/7527` 跑一遍关键链路。

为什么需要：`snapshot/parsed` 是本地采集（不入库），公开克隆里没有它。
如果代码偷偷依赖那里，公开出去的仓库就是"跑不起来"的 —— 这个脚本专门抓这种情况。

做三件事（本地采集目录临时改名，结束后**无论如何**恢复）：
  1. 数据来源判定（`engine.paths.source_label()` 应显示 `scenario/7527`）
  2. 引擎自检 `tools/selftest_engine.py`
  3. 计分对账 `tools/hex_score_audit.py`（要求 exit 0：引擎 remaining == 平台 354/234）
  4. 真实想定单局 `env/run_episode.py --agent greedy-multi --snapshot scenario/7527`

用法：py -3.13 tools\\verify_public_clone.py [--keep]      # --keep 不恢复（调试用）
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "snapshot", "parsed")
HIDDEN = os.path.join(ROOT, "snapshot", "parsed.__hidden_for_test")
PUBLISHED = os.path.join(ROOT, "scenario", "7527")
PY = sys.executable


def run(rel, argv=(), env_extra=None):
    env = dict(os.environ, PYTHONUTF8="1")
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([PY, rel] + list(argv), cwd=ROOT, capture_output=True,
                       text=True, env=env)
    tail = (p.stdout or "").strip().splitlines()[-1:] or \
           (p.stderr or "").strip().splitlines()[-1:]
    return p.returncode, (tail[0] if tail else "")[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="不恢复（调试用）")
    args = ap.parse_args()

    if not os.path.isdir(PUBLISHED):
        print("缺少 %s：先跑 py -3.13 tools\\publish_scenario.py" % PUBLISHED)
        return 1
    hidden = False
    try:
        if os.path.isdir(LOCAL):
            os.rename(LOCAL, HIDDEN)
            hidden = True
        print("本地采集 %s" % ("已临时隐藏（模拟公开克隆）" if hidden else "本来就不存在"))
        results = []
        rc, out = run("-c", [])
        # 数据来源
        p = subprocess.run([PY, "-c",
                            "import sys;sys.path.insert(0,'.');"
                            "from engine import paths;print(paths.source_label())"],
                           cwd=ROOT, capture_output=True, text=True)
        label = (p.stdout or "").strip()
        print("数据来源：%s" % label)
        results.append(("数据来源回退到 scenario/7527", "scenario/7527" in label))
        for name, rel, argv in (
                ("引擎自检", "tools/selftest_engine.py", ()),
                ("计分对账（引擎==平台 354/234）", "tools/hex_score_audit.py", ()),
                ("真实想定单局", "env/run_episode.py",
                 ("--agent", "greedy-multi", "--episodes", "1",
                  "--snapshot", "scenario/7527"))):
            rc, tail = run(rel, argv)
            print("  %s %-26s exit=%d %s" % ("✅" if rc == 0 else "❌", name, rc, tail))
            results.append((name, rc == 0))
        bad = [n for n, ok in results if not ok]
        print("\n公开克隆自检：%d/%d 通过%s"
              % (len(results) - len(bad), len(results),
                 ("，未通过：%s" % bad) if bad else " ✅"))
        return 1 if bad else 0
    finally:
        if hidden and not args.keep and os.path.isdir(HIDDEN):
            os.rename(HIDDEN, LOCAL)
            print("已恢复 %s" % LOCAL)


if __name__ == "__main__":
    sys.exit(main())
