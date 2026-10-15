# -*- coding: utf-8 -*-
"""差分校验：`legal_actions(deep_validate=False)`（快速）是否与
`deep_validate=True`（参考口径：每个候选一次整状态事务校验）**逐状态等价**。

为什么需要它：快速路径不再对每个候选做整状态深拷贝（真实地图上那是 13 秒/次），
改为对 `READONLY_TYPES`（机动/等待/停止）做只读判定。这必须用证据守住，
而 CI 里没有真实地图（`snapshot/` 不入库），所以本工具可**在真实地图上本地跑**。

同时校验三个不变量：
  1. 两次枚举的**动作集合完全相同**（集合语义，顺序不计）；
  2. 枚举**只读**：状态指纹与随机数流状态前后一致；
  3. 抽查的候选**确实被 `step()` 接受**（含机动耗时一致）。

用法：
    py -3.13 tools/hex_env_diff.py                        # 合成 demo，20 步
    py -3.13 tools/hex_env_diff.py --snapshot snapshot/parsed --radius 2 --steps 8
"""
import argparse
import copy
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.hexwar_env import HexwarEnv        # noqa: E402

MAX_SECONDS = 900


def fingerprint(env):
    """状态指纹：位置/兵力/压制/姿态/弹药/时钟 + 随机数流。"""
    g = env.game
    units = sorted([u.id, u.pos, u.pos2, u.blood, u.keep, u.ste, u.att,
                    json.dumps(sorted((u.ammo or {}).items())),
                    json.dumps(u.detail, sort_keys=True, default=str)]
                   for u in g.units.values())
    return json.dumps({"second": g.second, "units": units,
                       "flags": [c.get("flag") for c in g.cities],
                       "rng": repr(g.rng.getstate()), "log": len(g.log)},
                      sort_keys=True, default=str)


def key(act):
    return json.dumps(act, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "samples", "demo"))
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20, help="比较多少个决策点")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", default="BLUE", choices=["BLUE", "RED"])
    ap.add_argument("--sample-check", type=int, default=12,
                    help="每个决策点抽查多少个候选确实被 step 接受")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    env = HexwarEnv(snapshot=args.snapshot, side=args.side,
                    max_seconds=MAX_SECONDS, fog=True)
    env.reset(seed=args.seed)
    rng = random.Random(args.seed + 7)

    lines = []
    lines.append("# 枚举差分校验（快速 vs 参考口径）")
    lines.append("")
    lines.append("- 快照：`%s`" % args.snapshot)
    lines.append("- 机动半径：%d　最大步数：%d　种子：%d　阵营：%s"
                 % (args.radius, args.steps, args.seed, args.side))
    lines.append("")

    mismatches = []
    readonly_bad = []
    accept_bad = []
    seconds_bad = []
    compared = 0
    sizes = []

    for i in range(args.steps):
        if env.done:
            break
        before = fingerprint(env)
        fast = env.legal_actions(move_radius=args.radius, deep_validate=False)
        after_fast = fingerprint(env)
        if before != after_fast:
            readonly_bad.append(i)
        deep = env.legal_actions(move_radius=args.radius, deep_validate=True)
        if fingerprint(env) != before:
            readonly_bad.append(i)
        compared += 1
        sizes.append((len(fast), len(deep)))
        fs, ds = {key(a) for a in fast}, {key(a) for a in deep}
        if fs != ds:
            mismatches.append({"step": i,
                               "only_fast": sorted(fs - ds)[:4],
                               "only_deep": sorted(ds - fs)[:4]})

        # 抽查：快速枚举给的动作必须真的被 step 接受（在克隆上试，不动原状态）
        for act in rng.sample(fast, min(args.sample_check, len(fast))) if fast else []:
            probe = copy.deepcopy(env)
            act_copy = copy.deepcopy(act)
            _, _, _, info = probe.step(act_copy)
            if info.get("invalid_action"):
                accept_bad.append((i, act, info.get("msg")))
            elif act["type"] == "move" and act.get("seconds") != info.get("seconds"):
                seconds_bad.append((i, act.get("seconds"), info.get("seconds")))

        if not fast:
            break
        env.step(rng.choice(fast))

    lines.append("## 结果")
    lines.append("")
    lines.append("- 比较决策点：**%d**" % compared)
    lines.append("- 动作数（快速 / 参考）：%s" % (sizes,))
    lines.append("- 动作集合不等价：**%d**" % len(mismatches))
    lines.append("- 枚举非只读（状态/随机数流被改）：**%d**" % len(readonly_bad))
    lines.append("- 抽查候选被 step 拒绝：**%d**" % len(accept_bad))
    lines.append("- 抽查机动耗时不一致：**%d**" % len(seconds_bad))
    lines.append("")
    for m in mismatches[:5]:
        lines.append("- ⚠ 决策点 %d：仅快速有 %s；仅参考有 %s"
                     % (m["step"], m["only_fast"], m["only_deep"]))
    for a in accept_bad[:5]:
        lines.append("- ⚠ 决策点 %d：step 拒绝 %s → %s" % a)
    for s in seconds_bad[:5]:
        lines.append("- ⚠ 决策点 %d：枚举耗时 %s vs step 耗时 %s" % s)
    ok = not (mismatches or readonly_bad or accept_bad or seconds_bad)
    lines.append("")
    lines.append("**结论：%s**" % ("等价 ✅" if ok else "存在分歧 ❌"))

    report = args.report or os.path.join(ROOT, "analysis",
                                         "env_diff_%s.md"
                                         % os.path.basename(args.snapshot.rstrip("\\/")))
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print("\n报告：%s" % report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
