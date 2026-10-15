# -*- coding: utf-8 -*-
"""规则强化训练环境 —— 统一入口（train/ 自包含副本内使用）。

这套环境的**唯一想定就是比赛初设**：想定 7527「城市攻防作战（2V2）」的真实地图、
编成与夺控点（随仓库公开在 `../scenario/7527/`，来历与权利说明见那里的 README）。
配置不写 `snapshot` 时自动用它；本环境**不含任何合成 demo**。

为什么需要这个入口：**规则就是奖励**。这套环境的所有分值/战果都来自
`engine/`（已用 86 条线上实测裁决对账到零残差）。只要引擎被改坏，
训练出来的策略就是在学错的规则 —— 所以本入口把「**先过规则门，再开训**」
做成默认行为：

    py -3.13 run_train.py train --algo ppo --config configs/ppo_realmap.json

子命令
------
  rules       规则符合性门：selftest + verify_tables/rules/extra/unfavorable + 计分对账
  rules --live  再加上「线上实测裁决逐条对账」（需要 archive/exploration 里的样本，没样本自动跳过）
  selftest    引擎自检（几何/通视/机动/裁决/夺控）
  env         单局跑一遍环境（看渲染 / 摸动作空间）
  mc          自博弈蒙特卡洛（env.selfplay）
  ppo         神经策略训练（env.neural_train）
  eval        策略评测（env.policy_eval）
  train       **推荐**：先跑规则门，过了再按 --algo 训练
  diff        枚举等价性差分（快口径 vs 深度校验）
  check       train/ 与核心仓库的漂移检查（MANIFEST.json）

约定
----
* 配置是 JSON（`configs/*.json`），本文件里每个子命令都写清**允许哪些键**；
  配置里出现未支持的键会**报错退出**，不静默忽略（免得你以为参数生效了）。
  键名以 `note` / `_` 开头的当文档字段（写参数来历、对比基线）。
* 相对路径按 `train/` 解析；`snapshot` 默认 `../scenario/7527`（比赛初设）。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ★ 训练环境只有**比赛初设**：想定 7527「城市攻防作战（2V2）」的真实地图与编成，
#   随仓库分发在 ../scenario/7527/（工具 tools/publish_scenario.py，出处见其 README）。
#   配置里不写 snapshot 时就用它 —— 不提供任何合成 demo。
DEFAULT_SCENARIO = "../scenario/7527"

# 每条规则门 = (显示名, [脚本相对 train/ 的路径])
RULES_GATE = [
    ("裁决表", "tools/verify_tables.py"),
    ("推演规则", "tools/verify_rules.py"),
    ("新增规则", "tools/verify_rules_extra.py"),
    ("不利修正", "tools/verify_unfavorable.py"),
    ("计分对账", "tools/hex_score_audit.py"),
]
RULES_LIVE = [
    ("线上裁决对账", "tools/verify_live_crttables.py"),
    ("线上裁决页对账", "tools/verify_live_caijue.py"),
]

# 每个子命令允许的配置键 → CLI 参数
FLAGS = {
    "snapshot": "--snapshot",
    "allow_real_map": "--allow-real-map",
    "move_radius": "--move-radius",
    "deep_validate": "--deep-validate",
    "episodes": "--episodes",
    "scoring": "--scoring",
    "seed": "--seed",
    "train_seed_start": "--train-seed-start",
    "eval_seeds": "--eval-seeds",
    "max_seconds": "--max-seconds",
    "max_steps": "--max-steps",
    "epsilon": "--epsilon",
    "batch_episodes": "--batch-episodes",
    "buckets": "--buckets",
    "hidden_dim": "--hidden-dim",
    "learning_rate": "--learning-rate",
    "observation_schema": "--observation-schema",
    "baseline_offset": "--baseline-offset",
    "update_mode": "--update-mode",
    "ppo_clip": "--ppo-clip",
    "update_epochs": "--update-epochs",
    "checkpoint": "--checkpoint",
    "resume": "--resume",
    "report": "--report",
    "agent": "--agent",
    "side": "--side",
    "seeds": "--seeds",
    "radius": "--radius",
    "steps": "--steps",
    "render": "--render",
    "no_fog": "--no-fog",
}
ALLOWED = {
    "run_episode": ("snapshot", "agent", "episodes", "side", "seed", "max_seconds",
                    "max_steps", "move_radius", "deep_validate", "scoring",
                    "no_fog", "render"),
    "mc": ("snapshot", "allow_real_map", "move_radius", "deep_validate", "episodes",
           "scoring", "seed", "epsilon", "train_seed_start", "eval_seeds",
           "max_seconds", "max_steps", "checkpoint", "resume", "report"),
    "ppo": ("snapshot", "allow_real_map", "move_radius", "deep_validate", "episodes",
            "seed", "train_seed_start", "eval_seeds", "batch_episodes", "max_seconds",
            "max_steps", "buckets", "hidden_dim", "learning_rate",
            "observation_schema", "baseline_offset", "update_mode", "scoring",
            "ppo_clip", "update_epochs", "checkpoint", "resume", "report"),
    "eval": ("checkpoint", "snapshot", "seeds", "report", "allow_real_map"),
    "diff": ("snapshot", "radius", "steps"),
}
MODULE = {"mc": "env.selfplay", "ppo": "env.neural_train", "eval": "env.policy_eval",
          "run_episode": "env.run_episode", "diff": "tools.hex_env_diff"}


def load_cfg(path, sub):
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    # `note` / `_xxx` 是**文档键**（写参数来历、对比基线），不当作 CLI 参数
    cfg = {k: v for k, v in cfg.items()
           if k != "note" and not k.startswith("_")}
    allowed = ALLOWED[sub]
    bad = sorted(set(cfg) - set(allowed))
    if bad:
        raise SystemExit("配置 %s 里有 %s 不支持的键：%s（允许：%s）"
                         % (path, sub, bad, list(allowed)))
    return cfg


def to_argv(cfg):
    argv = []
    for k, v in cfg.items():
        flag = FLAGS[k]
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv += [str(x) for x in v]
        elif v is not None:
            argv += [flag, str(v)]
    return argv


def run(cmd, title=None):
    if title:
        print("\n$ %s" % title)
    print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=HERE).returncode
    if rc:
        print("!! 退出码 %d" % rc)
    return rc


def py_script(rel, argv):
    os.makedirs(os.path.join(HERE, "analysis"), exist_ok=True)
    return [PY, rel] + argv


def py_module(name, argv):
    os.makedirs(os.path.join(HERE, "analysis"), exist_ok=True)
    return [PY, "-m", name] + argv


def ensure_dirs(cfg):
    for key in ("checkpoint", "report"):
        p = cfg.get(key)
        if p:
            os.makedirs(os.path.dirname(os.path.join(HERE, p)) or HERE, exist_ok=True)


def rules_gate(live=False):
    fails = []
    for name, rel in RULES_GATE + (RULES_LIVE if live else []):
        if not os.path.exists(os.path.join(HERE, rel)):
            print("  跳过 %-14s（%s 不存在）" % (name, rel))
            continue
        p = subprocess.run([PY, rel], cwd=HERE, capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-1:] or \
               (p.stderr or "").strip().splitlines()[-1:]
        ok = p.returncode == 0
        print("  %s %-14s %s" % ("✅" if ok else "❌", name, tail))
        if not ok:
            fails.append((name, rel, tail))
    if fails:
        print("\n规则门未通过：%s" % [f[0] for f in fails])
        print("⇒ 规则不对就别开训（否则学的是错规则）。先修引擎/表，再看上面各脚本的输出。")
        return 1
    print("\n规则门通过 ✅（引擎与已实测规则一致）")
    return 0


def main():
    ap = argparse.ArgumentParser(description="规则强化训练入口（先过规则门再开训）")
    ap.add_argument("cmd", choices=("rules", "selftest", "env", "mc", "ppo", "eval",
                                    "train", "diff", "check"))
    ap.add_argument("--config")
    ap.add_argument("--algo", choices=("mc", "ppo"), default="ppo")
    ap.add_argument("--live", action="store_true", help="rules 时加上线上样本对账")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令行，不执行")
    args, extra = ap.parse_known_args()

    if args.cmd == "rules":
        return rules_gate(live=args.live)
    if args.cmd == "selftest":
        return run(py_script("tools/selftest_engine.py", []))
    if args.cmd == "check":
        return run([PY, os.path.join("..", "tools", "build_train_env.py"), "--check"])

    sub = {"env": "run_episode", "mc": "mc", "ppo": "ppo", "eval": "eval",
           "diff": "diff", "train": args.algo}[args.cmd]
    if args.cmd == "train":
        # ★ 规则强化训练的核心约定：**先过规则门，再开训**。
        #   规则就是奖励来源；引擎/表被改坏时训练出来的策略在学错规则。
        rc = rules_gate(live=args.live)
        if rc:
            return rc
    cfg = load_cfg(args.config, sub)
    if sub in ("mc", "ppo", "eval", "diff") and "snapshot" not in cfg \
            and "--snapshot" not in extra:
        cfg["snapshot"] = DEFAULT_SCENARIO      # 比赛初设，没有别的样例
    if sub == "eval" and not (cfg.get("seeds") or "--seeds" in extra):
        raise SystemExit("eval 需要 --seeds（或在配置里给 seeds）")
    if sub in ("mc", "ppo") and not (cfg.get("checkpoint") or "--checkpoint" in extra):
        raise SystemExit("%s 需要 checkpoint（或在配置里给 checkpoint）" % sub)
    argv = to_argv(cfg) + extra
    cmd = py_module(MODULE[sub], argv)
    ensure_dirs(cfg)
    if args.dry_run:
        print("$ " + " ".join(cmd))
        return 0
    return run(cmd)


if __name__ == "__main__":
    sys.exit(main())
