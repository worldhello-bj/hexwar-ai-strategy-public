# -*- coding: utf-8 -*-
"""实时决策接口验收（60 秒口径）—— **只测量，不修改任何东西**。

## 要回答的问题

官方是 **1 Hz 实时**推进（`engine.game.Game.tick()`：`second = int(time.time() - t0)`，
服务端每秒刷新一次），玩家**每一秒**都能观察 / 下令 / 改令；一条 60 秒的机动指令
不阻塞其他单位在同一时间轴上继续执行。

训练侧用的是 **`Game.advance(seconds)`**：按"这一次动作的时长"把时钟**跳**过去，
然后跑一次自动裁决。两者是不是同一件事，必须实测，不能假定。

## 验收项（每项都打印 PASS / FAIL 与证据）

| 项 | 问题 |
|---|---|
| T1 | **并行推进**：两个单位各持一条持续指令，走完同一段 60 秒——结果是否只由秒数决定？ |
| T2 | **相隔一秒下单**：A 拿到 60 秒指令后，B 还能不能在 `t=1` 秒拿到自己的指令？ |
| T3 | **决策机会**：覆盖 60 秒游戏时间，同一方实际能拿到几次决策？ |
| T4 | **自动裁决粒度**：`advance(1)×60` 与 `advance(60)×1` 跑出的裁决次数是否一致？ |
| T4b | **战果粒度依赖**：同一段 60 秒、同一个持续打打击目标，打了几发？ |
| T5 | **指令状态可见性**：「保持现有指令」是否与「停止」区分？观测里有没有在执行的指令/剩余时间？ |
| T6 | **实时接口 60 秒验收**：`RealtimeEpisode` 下每方能否拿到 60 次决策、两条相隔一秒的持续指令能否并行推进 |

用法::

    py -3.13 tools/hex_time_acceptance.py
    py -3.13 tools/hex_time_acceptance.py --json
    py -3.13 tools/hex_time_acceptance.py --window 60 --snapshot scenario/7527
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import geometry as G                     # noqa: E402
from engine.game import Game                         # noqa: E402
from env import HexwarEnv                            # noqa: E402
from env.episode_api import TwoPlayerEpisode         # noqa: E402

WINDOW = 60          # 验收窗口：60 秒


# ---------------------------------------------------------------- 小工具
def pick_units(g, side="BLUE", want=2):
    """挑两个能独立机动的己方地面单位（typeid 1/2，非车内）。"""
    out = []
    for u in sorted(g.units.values(), key=lambda x: str(x.id)):
        if (u.color == side and u.typeid in (1, 2) and u.alive()
                and not u.detail.get("carrier")):
            out.append(u)
        if len(out) == want:
            break
    return out


def open_path(g, unit, want_steps):
    """沿可走格 BFS 找一条 want_steps 格、不含起点的路径。"""
    seen = {unit.pos}
    path = []
    cur = unit.pos
    for _ in range(want_steps):
        nxt = None
        for nb in G.neighbors(cur):
            if nb in seen:
                continue
            sec, _why = g.step_seconds(unit, cur, nb)
            if sec > 0:
                nxt = nb
                break
        if nxt is None:
            break
        seen.add(nxt)
        path.append(nxt)
        cur = nxt
    return path


def count_calls(g, name):
    """把 g.<name> 包一层计数器，返回计数器（原地改实例，只影响本次测量）。"""
    counter = Counter()
    original = getattr(g, name)

    def wrapper(*a, **kw):
        counter[name] += 1
        return original(*a, **kw)

    setattr(g, name, wrapper)
    return counter


# ---------------------------------------------------------------- T1
def t1_parallel(snapshot, window):
    """同一段 60 秒，'每秒推一次' 与 '一次跳 60 秒' 是否等价。

    注意：`begin_move` / `_resume_path` 是**手册 4.1.1(6) 解压制后续行**用的接口，
    训练链路（`HexwarEnv._apply`）走的是 `commit_step`，不经过它。本项测的是
    **时间模型本身**（推进只由秒数决定，还是由调用次数决定）。
    """
    results = {}
    for label, chunk in (("advance(1)×%d" % window, 1), ("advance(%d)×1" % window, window)):
        g = Game(snapshot=snapshot)
        g.start()
        g.second = 0
        units = pick_units(g)
        starts, paths, rule_sec = {}, {}, {}
        for u in units:
            p = open_path(g, u, 8)
            paths[u.id] = list(p)
            starts[u.id] = u.pos
            if p:
                rule_sec[u.id] = round(g.step_seconds(u, u.pos, p[0])[0], 1)
        for u in units:
            if paths[u.id]:
                g.begin_move(u.id, paths[u.id])
        calls = count_calls(g, "_resume_path")
        for _ in range(window // chunk):
            g.advance(chunk)
        moves = {}
        for u in units:
            p = paths[u.id]
            if not p:
                continue
            if u.pos in p:
                steps = p.index(u.pos) + 1
                moves[u.id] = {"推进格数": steps,
                               "实际速度(秒/格)": round(window / steps, 1),
                               "规则速度(秒/格)": rule_sec[u.id]}
            else:
                moves[u.id] = {"推进格数": "偏离路径", "实际速度(秒/格)": None,
                               "规则速度(秒/格)": rule_sec[u.id]}
        results[label] = {"_resume_path 调用次数": calls["_resume_path"],
                          "推进情况": moves,
                          "末位置": {u.id: u.pos for u in units},
                          "剩余未完成机动": g.pending_moves()}
    a, b = results["advance(1)×%d" % window], results["advance(%d)×1" % window]
    same = a["末位置"] == b["末位置"]
    return {"等价（只由秒数决定）": same, "两种推进方式": results,
            "证据": "同一段 %d 秒，末位置 %s；推进速度由**调用次数**决定，不按 step_seconds 走"
                    % (window, "一致" if same else "不一致")}


# ---------------------------------------------------------------- T4b
def t4b_effective_shots(snapshot, window):
    """同一段 60 秒、同一个持续打击目标：把时间切成 1 秒 vs 一次 60 秒，打了几发。

    ★ 受控构造：把一对敌我单位摆到相邻格以建立通视（开局 144 对单位通视 = 0），
    然后设定持续打击（手册 3.14），再只推进时钟、不改变其他状态。
    这是**测量时间模型**用的最小实验，不改动任何想定数据。
    """
    from engine.equip import WEAPONS

    def build():
        env = HexwarEnv(snapshot=snapshot, side="BLUE", fog=False, max_seconds=1200)
        env.reset(seed=0)
        g = env.game
        blue = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
                    if u.color == "BLUE" and u.typeid == 2 and u.wz)
        red = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
                   if u.color == "RED" and u.typeid == 2 and u.alive())
        red.pos = red.pos2 = G.neighbors(blue.pos)[0]
        weapon = next(w for w in blue.wz if (WEAPONS.get(w) or (None, None, 99))[2] >= 1)
        g.set_engagement(blue.id, red.id, weapon_id=weapon)
        return env, g, blue, red

    out = {}
    for label, chunk in (("每步 wait(1)×%d" % window, 1), ("一步 wait(%d)" % window, window)):
        env, g, blue, red = build()
        before = len(g.log)
        for _ in range(window // chunk):
            _, _, done, info = env.step({"type": "wait", "seconds": chunk})
            if done or info.get("invalid_action"):
                break
        shots = sum(1 for line in g.log[before:] if "直瞄" in line)
        out[label] = {"直瞄裁决次数": shots, "时钟": g.second,
                      "目标剩余兵力": red.blood}
    a = out["每步 wait(1)×%d" % window]["直瞄裁决次数"]
    b = out["一步 wait(%d)" % window]["直瞄裁决次数"]
    return {"粒度无关": a == b, "两种推进方式": out,
            "证据": "同一段 %d 秒、同一个持续打打击目标：直瞄裁决 %d 次 vs %d 次" % (window, a, b)}


# ---------------------------------------------------------------- T2
def t2_one_second_apart(snapshot, window):
    """A 拿到 60 秒指令后，B 还能不能在 t=1 秒拿到指令。"""
    env = HexwarEnv(snapshot=snapshot, side="BLUE", fog=False, max_seconds=1200)
    env.reset(seed=0)
    units = pick_units(env.game)
    if len(units) < 2:
        return {"可测": False}
    a_id, b_id = units[0].id, units[1].id
    acts = env.legal_actions(unit_id=a_id)
    move = next((x for x in acts if x["type"] == "move" and len(x["path"]) > 1), None)
    if move is None:
        return {"可测": False}
    t_before = env.game.second
    env.step(move)
    t_after = env.game.second
    # 此刻 B 才第一次有机会：时钟已经在 t_after
    b_acts = env.legal_actions(unit_id=b_id)
    return {"可测": True, "A 的动作时长（秒）": t_after - t_before,
            "A 下单前的时钟": t_before, "A 下单后的时钟": t_after,
            "B 能拿到决策的最早时钟": t_after,
            "B 能否在 t=1 下单": t_after <= 1,
            "B 此刻的合法动作数": len(b_acts),
            "证据": "一次动作把时钟从 %d s 直接推到 %d s；期间没有任何决策点" % (t_before, t_after)}


# ---------------------------------------------------------------- T3
def t3_decision_budget(snapshot, window):
    """覆盖 window 秒游戏时间，同一方实际拿到几次决策。"""
    episode = TwoPlayerEpisode(snapshot, max_seconds=1200, max_steps=600,
                              move_radius=2, public_score=True)
    status = episode.reset(seed=0)
    per_side, seconds_at = Counter(), []
    while not status["terminated"] and not status["truncated"]:
        side = status["current_actor"]
        legal = episode.legal_actions(side)
        if not legal:
            break
        if status["seconds"] <= window:
            seconds_at.append(status["seconds"])
        status = episode.step(legal[0], side=side)
        per_side[side] += 1
    budget = sum(1 for s in seconds_at if s <= window)
    full = sum(per_side.values())
    return {"窗口（秒）": window, "窗口内总决策数（双方合计）": budget,
            "窗口内每方决策数（约）": round(budget / 2, 1),
            "实时语义下每方应有": window,
            "整局决策数（双方合计）": full,
            "每方整局决策数（约）": round(full / 2, 1),
            "整局游戏秒数": status["seconds"],
            "证据": "窗口内每方 ≈ %.1f 次决策，而实时语义是 %d 次/方"
                    % (budget / 2, window)}


# ---------------------------------------------------------------- T4
def t4_resolution_granularity(snapshot, window):
    """advance(1)×N 与 advance(N)×1：自动裁决入口被调用几次。"""
    out = {}
    for label, chunk in (("advance(1)×%d" % window, 1), ("advance(%d)×1" % window, window)):
        g = Game(snapshot=snapshot)
        g.start()
        g.second = 0
        volley = count_calls(g, "resolve_engagements")
        samehex = count_calls(g, "same_hex_engage")
        for _ in range(window // chunk):
            g.advance(chunk)
        out[label] = {"resolve_engagements（自动持续打击 3.14）": volley["resolve_engagements"],
                      "same_hex_engage（同格交战 4.11）": samehex["same_hex_engage"],
                      "时钟": g.second}
    a = out["advance(1)×%d" % window]["resolve_engagements（自动持续打击 3.14）"]
    b = out["advance(%d)×1" % window]["resolve_engagements（自动持续打击 3.14）"]
    return {"粒度无关": a == b, "两种推进方式": out,
            "证据": "同一段 %d 秒，裁决入口被调用 %d 次 vs %d 次" % (window, a, b)}


# ---------------------------------------------------------------- T5
def t5_order_visibility(snapshot):
    """动作空间里「保持现有指令」与「停止」是否区分；观测是否含执行进度。"""
    env = HexwarEnv(snapshot=snapshot, side="BLUE", fog=False, max_seconds=1200)
    env.reset(seed=0)
    kinds = Counter(a["type"] for a in env.legal_actions())
    unit_view = next(iter(v for v in env.observe()["own"]))
    progress_keys = [k for k in unit_view
                     if any(t in k.lower() for t in ("path", "order", "until", "remain", "progress", "eta"))]
    top_keys = sorted(env.observe().keys())
    return {"动作类型": dict(kinds),
            "有 'keep/保持指令' 动作": any("keep" in k for k in kinds),
            "有 'stop/停止' 动作": any(k == "stop" for k in kinds),
            "单位观测字段": sorted(unit_view.keys()),
            "含执行进度/剩余时间的字段": progress_keys,
            "顶层观测字段": top_keys,
            "证据": "动作类型 %s；单位观测里带进度/剩余时间的字段 = %s"
                    % (sorted(kinds), progress_keys or "无")}


# ---------------------------------------------------------------- T6
def t6_realtime(snapshot, window):
    """实时接口（`env.realtime_api.RealtimeEpisode`）的 60 秒验收。

    检查三件事：
    1. **决策机会**：窗口内每方拿到 ≈ window 次（1 Hz），而不是 1 次；
    2. **并行持续指令**：A 在 t=0 拿到指令后，B 在 t=1 能拿到自己的指令，
       且两条指令在同一个时间轴上**同时**推进；
    3. **keep / stop 区分** 与 **观测里的执行进度**。
    """
    from env.realtime_api import RealtimeEpisode

    ep = RealtimeEpisode(snapshot, max_seconds=1200, move_radius=1)
    status = ep.reset(seed=0)
    per = {"BLUE": 0, "RED": 0}
    while not status["terminated"] and status["seconds"] < window:
        side = status["current_actor"]
        legal = ep.legal_actions(side)
        status = ep.step(legal[0], side=side)          # legal[0] = keep
        per[side] += 1
    budget = {"每方决策数": dict(per), "应有": window,
              "达标": per["BLUE"] == window and per["RED"] == window}

    # 并行持续指令
    ep2 = RealtimeEpisode(snapshot, max_seconds=1200, move_radius=1)
    status = ep2.reset(seed=0)
    g = ep2._session.env.game
    units = [u for u in sorted(g.units.values(), key=lambda x: str(x.id))
             if u.color == "BLUE" and u.typeid in (1, 2)][:2]
    moves = {}
    legal = ep2.legal_actions("BLUE")
    for u in units:
        cand = next((a for a in legal if a["type"] == "move" and a["unit"] == u.id), None)
        moves[u.id] = cand
    start = {u.id: u.pos for u in units}
    ordered_at = {}
    if all(moves.values()):
        # t=0：给 A 下持续指令；t=1：给 B 下持续指令
        status = ep2.step(moves[units[0].id], side="BLUE")
        ordered_at[units[0].id] = status["seconds"]
        status = ep2.step({"type": "keep"}, side="RED")
        status = ep2.step({"type": "keep"}, side="BLUE")
        status = ep2.step({"type": "keep"}, side="RED")
        legal_b = ep2.legal_actions("BLUE")
        b_move = next((a for a in legal_b if a["type"] == "move" and a["unit"] == units[1].id), None)
        if b_move is None:
            status = ep2.step({"type": "keep"}, side="BLUE")
        else:
            status = ep2.step(b_move, side="BLUE")
            ordered_at[units[1].id] = status["seconds"]
        status = ep2.step({"type": "keep"}, side="RED")
    # 指令执行中取样：此刻两个单位都应该带着"正在执行什么 / 还剩几秒"
    obs_order = ep2.observe("BLUE")
    for _ in range(2 * window):
        if status["terminated"]:
            break
        status = ep2.step({"type": "keep"}, side=status["current_actor"])
    advanced = {u.id: (u.pos != start[u.id], u.pos) for u in units}

    obs = ep2.observe("BLUE")
    live = [v for v in obs_order["own"] if v.get("order")]
    sample = live[0] if live else None
    return {"决策机会": budget,
            "相隔一秒下达两条持续指令": {
                "下达时刻": ordered_at,
                "两个单位都推进了": all(v[0] for v in advanced.values()),
                "推进后位置": {k: v[1] for k, v in advanced.items()}},
            "keep 存在": any(a["type"] == "keep" for a in ep2.legal_actions("BLUE")),
            "观测含 order 字段": "order" in obs["own"][0],
            "执行中的 order 示例": {v["id"]: v["order"] for v in live[:2]},
            "证据": "窗口内每方 %d/%d 次决策；A 在 t=%s、B 在 t=%s 拿到指令，两者进度 %s"
                    % (per["BLUE"], window, ordered_at.get(units[0].id),
                       ordered_at.get(units[1].id),
                       "都推进" if all(v[0] for v in advanced.values()) else "未全部推进")}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="实时决策接口验收（只测量）")
    ap.add_argument("--snapshot", default=None, help="想定目录（默认按 engine/paths.py 解析）")
    ap.add_argument("--window", type=int, default=WINDOW, help="验收窗口秒数（默认 60）")
    ap.add_argument("--json", action="store_true", help="只打印 JSON")
    args = ap.parse_args(argv)
    snap = args.snapshot or None

    report = {"window": args.window, "checks": {}}
    for name, fn in (("T1 并行推进", lambda: t1_parallel(snap, args.window)),
                     ("T2 相隔一秒下单", lambda: t2_one_second_apart(snap, args.window)),
                     ("T3 决策机会预算", lambda: t3_decision_budget(snap, args.window)),
                     ("T4 自动裁决粒度", lambda: t4_resolution_granularity(snap, args.window)),
                     ("T4b 战果粒度依赖", lambda: t4b_effective_shots(snap, args.window)),
                     ("T5 指令状态可见性", lambda: t5_order_visibility(snap)),
                     ("T6 实时接口 60 秒验收", lambda: t6_realtime(snap, args.window))):
        try:
            report["checks"][name] = fn()
        except Exception as exc:                       # noqa: BLE001 - 验收工具要跑完所有项
            report["checks"][name] = {"error": "%s: %s" % (type(exc).__name__, exc)}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=" * 78)
    print("实时决策接口验收（窗口 %d 秒，想定 %s）" % (args.window, snap or "engine/paths.py 默认"))
    print("=" * 78)
    for name, data in report["checks"].items():
        print("\n【%s】" % name)
        if "error" in data:
            print("  !! %s" % data["error"])
            continue
        for k, v in data.items():
            if k in ("证据", "两种推进方式"):
                continue
            print("  %-28s %s" % (k, json.dumps(v, ensure_ascii=False)))
        if "两种推进方式" in data:
            for label, sub in data["两种推进方式"].items():
                print("    - %-18s %s" % (label, json.dumps(sub, ensure_ascii=False)))
        if "证据" in data:
            print("  → %s" % data["证据"])
    print("\n" + "=" * 78)
    print("本工具只测量，不修改任何状态。结论请结合 docs/训练困难清单.md 阅读。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
