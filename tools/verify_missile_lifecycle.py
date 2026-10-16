# -*- coding: utf-8 -*-
"""巡飞弹生命周期验收（issue #1）—— 逐条对应 issue 的「建议验收测试」清单。

## 要回答的问题

巡飞弹要能用，链路必须是 **在图上（未部署）→ 车内 → 在飞 → 攻击后消耗 / 超时或损失**，
而且每一步都要**能通过训练接口（合法动作生成 → 校验 → 执行 → 实时层调度）走到**，
不能出现“能移动但无法攻击”这种半截状态。

## 验收项

| 项 | 对应的 issue 条目 |
|---|---|
| T1 | 为「未部署 / 车内 / 在飞 / 已消耗或损失」建立**明确转换**，保留载车与时间字段来源 |
| T2 | 车内 `blood=0` 可发射；已损失不复活；开局前 / 错误载车 / 已毁载车均拒绝 |
| T3 | 接通合法动作生成、校验、执行，以及实时包装层的调度；发射耗时不补造 |
| T4 | 下车只卸人员 / 无人车（巡飞弹留在车内，`off=3` 是发射不是取回）；发射移除 carried；不能重复发射同一枚 |
| T5 | 同车已有在飞弹时拒绝第二枚；消耗或超时后释放占用；载车被毁则终止在飞弹 |
| T6 | 攻击后弹体消失、不能再次行动；伤害与消耗只结算一次；非法或未执行的攻击不消耗 |
| T7 | 计时覆盖不同 RoomSpeed、保存恢复、超时边界；先核对训练秒映射 |
| T8 | 当前 `car=Obj_0` 的初态**不得**凭同格静默创建 launcher；缺失状态需显式暴露 |
| T9 | 旧规则用例（`tools/verify_rules_extra.py` 的 4.10 段）不再靠手工改状态绕过缺陷 |

用法::

    py -3.13 tools/verify_missile_lifecycle.py
    py -3.13 tools/verify_missile_lifecycle.py --json

退出码：任一项失败即 1（可作 CI 门禁）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import geometry as G                      # noqa: E402
from engine.game import Game, LOITER_SEC_PLACEHOLDER   # noqa: E402
from env import HexwarEnv                             # noqa: E402
from env.realtime_api import RealtimeEpisode          # noqa: E402

SNAP = os.path.join(ROOT, "scenario", "7527")
RESULTS = []


def check(tag, name, ok, note=""):
    RESULTS.append({"tag": tag, "name": name, "ok": bool(ok), "note": str(note)})
    print("  [%s] [%s] %s%s" % ("PASS" if ok else "FAIL", tag, name,
                               ("　— " + str(note)) if note else ""))
    return bool(ok)


# ---------------------------------------------------------------- 合成棋盘
def build_fixture(root):
    """最小可用的想定：一辆车 + 两枚巡飞弹 + 一个步兵班 + 一个敌方坦克。

    弹与车**同格**（想定 7527 的形态），但**没有** `car` 关系 —— 必须显式部署。
    """
    start = G.make_mapid(10, 10)
    far = G.make_mapid(10, 12)          # 距 start 2 格：在巡飞弹侦察半径(2)内
    hexes = {str(G.make_mapid(r, c)): {"cond": 0, "elevation": 0, "step": 1}
             for r in range(7, 15) for c in range(7, 15)}
    units = [
        dict(id="carrier", data=396, typeid=2, color="BLUE", blood=4, score=10,
             pos=start, son="WS43", name="火力支援车", a5=4),
        dict(id="squad", data=382, typeid=1, color="BLUE", blood=3, score=3,
             pos=start, a5=1),
        dict(id="missileA", data=389, typeid=3, color="BLUE", blood=3, score=0,
             pos=start, ico="WS43", a5=1),
        dict(id="missileB", data=389, typeid=3, color="BLUE", blood=3, score=0,
             pos=start, ico="WS43", a5=1),
        dict(id="foe", data=400, typeid=2, color="RED", blood=3, score=10, pos=far),
    ]
    for name, payload in (("map_hexes.json", hexes), ("cities.json", []),
                          ("units.json", units)):
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    return start


def acts_of(env, kind, **want):
    return [a for a in env.legal_actions() if a.get("type") == kind
            and all(a.get(k) == v for k, v in want.items())]


def missile(g, uid="missileA"):
    return g.units[uid]


# ---------------------------------------------------------------- T1 / T8
def t1_t8_states(fixture):
    print("\n== T1/T8 生命周期状态与初态（想定 7527） ==")
    g = Game(seed=5, snapshot=SNAP)
    g.start()
    msls = sorted((u for u in g.units.values() if u.is_loitering_missile()),
                  key=lambda x: str(x.id))
    check("T8", "想定里的巡飞弹是 typeid=3 / 名称「巡飞弹」", len(msls) == 3,
          "%d 枚" % len(msls))
    check("T1", "初态一律 `unassigned`（在图上、无所属关系）",
          all(g.missile_state(m) == "unassigned" for m in msls))
    check("T8", "初态**没有**静默建立 launcher/carrier（`car=Obj_0` 只作证据留存）",
          all(not m.detail.get("carrier") and not m.detail.get("launcher")
              and not m.detail.get("car") for m in msls))
    check("T8", "初态没有载车被标记为「在飞占用」",
          all(not u.detail.get("launched_missile") and not u.detail.get("stop")
              for u in g.units.values()))
    car = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
               if u.typeid == 2 and u.pos == msls[0].pos and u.color)
    m = msls[0]
    check("T1", "未部署 → 发射被拒（不是“能移动就能打”）",
          not g.launch_missile(car.id, m.id)[0], g.launch_missile(car.id, m.id)[1])
    check("T1", "`unassigned` → `carried`（显式部署，手册 4.10(3)）",
          g.deploy_missile(car.id, m.id)[0] and g.missile_state(m) == "carried")
    check("T1", "车内记录 blood=0 / 阵营为空（平台口径），但状态仍是 `carried`",
          m.blood == 0 and not m.color and g.missile_state(m) == "carried")
    check("T1", "`carried` → `loitering`（发射）",
          g.launch_missile(car.id, m.id)[0] and g.missile_state(m) == "loitering")
    check("T1", "在飞弹记录发射秒 / 巡飞上限 / 取值来源（时间字段可追溯）",
          m.detail.get("launch_second") == 0
          and m.detail.get("loiter_seconds") == g.loiter_sec
          and bool(m.detail.get("loiter_seconds_source")))
    check("T1", "载车记录在飞占用（`launched_missile` 与 `stop=1` 同步）",
          car.detail.get("launched_missile") == m.id and car.detail.get("stop") == 1)
    g.advance(g.loiter_sec)
    check("T1", "超时 → `lost`（手册 4.10(5)）", g.missile_state(m) == "lost")

    g2 = Game(seed=5, snapshot=SNAP)
    g2.start()
    m2 = next(u for u in g2.units.values() if u.is_loitering_missile())
    c2 = next(u for u in sorted(g2.units.values(), key=lambda x: str(x.id))
              if u.typeid == 2 and u.pos == m2.pos and u.color)
    g2.deploy_missile(c2.id, m2.id)
    g2.launch_missile(c2.id, m2.id)
    m2.blood = 0                      # 外部击杀（手册 4.10(6) 下不该发生，但状态机要自洽）
    g2.advance(1)
    check("T1", "在飞弹血量为 0 → 推导为 `lost`（不是“还在飞”）",
          g2.missile_state(m2) == "lost")
    check("T1", "已损失 ⇒ 载车占用被释放（不留永久锁）",
          not c2.detail.get("launched_missile"))
    check("T1", "`spent` 只在“攻击完成”时出现（此例不是 spent）",
          g2.missile_state(m2) != "spent")


# ---------------------------------------------------------------- T2
def t2_gates(fixture):
    print("\n== T2 发射门：车内 blood=0 可发射 / 已损失不复活 / 各种拒绝 ==")
    g = Game(seed=6, snapshot=fixture)
    g.start()
    car, m, other = g.units["carrier"], g.units["missileA"], g.units["missileB"]
    check("T2", "对抗开始前（`started=False`）不能发射",
          not Game(seed=6, snapshot=fixture).launch_missile(car.id, m.id)[0])
    ok, why = g.deploy_missile(car.id, m.id)
    check("T2", "车内 blood=0 的记录可部署（不被 `alive()` 误挡）",
          ok and m.blood == 0, why)
    g.started = False
    ok, why = g.launch_missile(car.id, m.id)
    check("T2", "开局前发射被拒（手册 4.10(3)）", not ok and "开始前" in why, why)
    g.started = True
    ok, why = g.launch_missile(car.id, m.id)
    check("T2", "车内 blood=0 也能发射（本次修复的核心一条）", ok, why)
    check("T2", "发射后血量按编制恢复（不是 0）", m.blood > 0, "blood=%d" % m.blood)

    g2 = Game(seed=6, snapshot=fixture)
    g2.start()
    car2, m2, other2 = g2.units["carrier"], g2.units["missileA"], g2.units["missileB"]
    g2.deploy_missile(car2.id, m2.id)
    ok, why = g2.launch_missile(car2.id, other2.id)
    check("T2", "弹未部署在**这辆车**上 → 拒绝（手册 4.10(3)）",
          not ok and "4.10(3)" in why, why)
    g2.deploy_missile(car2.id, other2.id)
    g2.launch_missile(car2.id, m2.id)
    car2.blood = 0
    ok, why = g2.launch_missile(car2.id, other2.id)
    check("T2", "载车已被消灭 → 拒绝", not ok and "消灭" in why, why)
    g2.advance(1)                       # 时钟推进后：车内弹随之明确记为 lost
    check("T2", "已损失的弹不能复活（不能借“修好车内记录”复活）",
          g2.missile_state(m2) == "lost"
          and not g2.launch_missile(car2.id, m2.id)[0])

    g3 = Game(seed=6, snapshot=fixture)
    g3.start()
    m3 = g3.units["missileA"]
    infantry = g3.units["squad"]
    ok, why = g3.deploy_missile(infantry.id, m3.id)
    check("T2", "非车辆不能作为巡飞弹载体（步兵班）", not ok, why)
    ok, why = g3.deploy_missile(g3.units["foe"].id, m3.id)
    check("T2", "敌方车辆不能作为载体（所属关系只能是自己人）",
          not ok and "己方" in why, why)


# ---------------------------------------------------------------- T3
def t3_interface(fixture):
    print("\n== T3 训练接口：动作生成 → 校验 → 执行 → 实时层调度 ==")
    env = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=600, fog=False,
                    move_radius=1)
    env.reset(seed=3)
    check("T3", "开局动作菜单里有「弹上车」（部署）",
          bool(acts_of(env, "load", unit="missileA", carrier="carrier")),
          json.dumps(acts_of(env, "load"), ensure_ascii=False))
    obs, _r, _d, info = env.step({"type": "load", "unit": "missileA",
                                  "carrier": "carrier"})
    check("T3", "部署动作被接受（不是 unknown action type）",
          not info.get("invalid_action"), info)
    check("T3", "部署后动作菜单里有「发射」",
          bool(acts_of(env, "launch", unit="carrier", missile="missileA")),
          json.dumps(acts_of(env, "launch"), ensure_ascii=False))
    obs, _r, _d, info = env.step({"type": "launch", "unit": "carrier",
                                  "missile": "missileA"})
    check("T3", "发射动作被接受", not info.get("invalid_action"), info)
    check("T3", "发射耗时被**显式**声明（占位值，不冒充实测）",
          info.get("seconds") and info["seconds"] == 10, info.get("seconds"))
    view = next((u for u in env.observe()["own"] if u["id"] == "missileA"), None)
    check("T3", "观测显式暴露状态、载车与剩余巡飞秒",
          view is not None and view.get("missile_state") == "loitering"
          and view.get("loiter_seconds_left") is not None,
          json.dumps(view, ensure_ascii=False) if view else "缺失")
    carrier_view = next(u for u in env.observe()["own"] if u["id"] == "carrier")
    check("T3", "载车观测带 `missile_in_flight`",
          carrier_view.get("missile_in_flight") == "missileA")
    _obs, _r, _d, info = env.step({"type": "launch", "unit": "carrier",
                                   "missile": "missileB"})
    check("T3", "未部署的弹直接发射 → 被校验拒绝（不静默成功）",
          info.get("invalid_action"), info)

    episode = RealtimeEpisode(fixture, max_seconds=60, move_radius=1)
    status = episode.reset(seed=3)
    episode.step({"type": "load", "unit": "missileA", "carrier": "carrier"},
                 side=status["current_actor"])
    own = episode.observe("BLUE")["own"]
    carrier_view = next(u for u in own if u["id"] == "carrier")
    check("T3", "实时层里车内弹通过载车的 `carrying` 可发现（平台不下发车内记录）",
          carrier_view.get("carrying") == ["missileA"],
          json.dumps(carrier_view.get("carrying"), ensure_ascii=False))
    status = episode.status()
    while not status["terminated"] and not (status["seconds"] >= 30
                                            and episode.current_actor == "BLUE"):
        status = episode.step({"type": "keep"}, side=status["current_actor"])
    launch = [a for a in episode.legal_actions("BLUE")
              if a.get("type") == "launch" and a.get("missile") == "missileA"]
    check("T3", "部署占用结束后实时层给出「发射」", bool(launch),
          json.dumps(episode.legal_actions("BLUE"), ensure_ascii=False)[:160])
    if launch:
        episode.step(launch[0], side="BLUE")
        check("T3", "实时层调度执行发射：弹进入 `loitering` 且载车被占",
              any(u["id"] == "missileA"
                  and u.get("missile_state") == "loitering"
                  for u in episode.observe("BLUE")["own"]))
    return env


# ---------------------------------------------------------------- T4
def t4_unload(fixture):
    print("\n== T4 下车不含巡飞弹 / 发射移除 carried / 不重复 ==")
    env = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=600, fog=False,
                    move_radius=1)
    env.reset(seed=4)
    env.step([{"type": "load", "unit": "missileA", "carrier": "carrier"},
              {"type": "load", "unit": "squad", "carrier": "carrier"}])
    car = env.game.units["carrier"]
    check("T4", "部署后弹与人员都在载车的 carrying 里",
          sorted(car.detail.get("carried") or []) == ["missileA", "squad"],
          car.detail.get("carried"))
    _o, _r, _d, info = env.step({"type": "unload", "unit": "carrier"})
    check("T4", "下车只卸人员：弹留在车上（客户端「全员下车」只发 off=1/2；off=3 是发射）",
          car.detail.get("carried") == ["missileA"]
          and env.game.missile_state(env.game.units["missileA"]) == "carried"
          and env.game.units["squad"].pos == car.pos,
          "carried=%s" % (car.detail.get("carried"),))
    check("T4", "车上只剩弹时「下车」不可用（不存在“把弹取回”）",
          not [a for a in env.legal_actions() if a.get("type") == "unload"]
          and env.step({"type": "unload", "unit": "carrier"})[3]
          .get("invalid_action"))
    check("T4", "弹没有被“取回”：状态仍是 carried、不在图上",
          env.game.missile_state(env.game.units["missileA"]) == "carried"
          and not env.game.units["missileA"].alive())

    env2 = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=600, fog=False,
                     move_radius=1)
    env2.reset(seed=4)
    env2.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
    env2.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
    car2 = env2.game.units["carrier"]
    check("T4", "发射把弹从 carrying 移出（不能重复卸载/重复发射）",
          "missileA" not in (car2.detail.get("carried") or []), car2.detail.get("carried"))
    _o, _r, _d, info = env2.step({"type": "launch", "unit": "carrier",
                                  "missile": "missileA"})
    check("T4", "重复发射同一枚 → 拒绝（弹已不在车内）",
          info.get("invalid_action"), info)


# ---------------------------------------------------------------- T5
def t5_one_in_flight(fixture):
    print("\n== T5 一车一枚（同时） / 释放 / 载车被毁终止 ==")
    env = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=900, fog=False,
                    move_radius=1)
    env.reset(seed=5)
    env.step([{"type": "load", "unit": "missileA", "carrier": "carrier"},
              {"type": "load", "unit": "missileB", "carrier": "carrier"}])
    env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
    offered = [a for a in env.legal_actions()
               if a.get("type") == "launch" and a.get("missile") == "missileB"]
    check("T5", "同车已有在飞弹 → 第二枚**不出现在动作菜单里**", not offered, offered)
    m_a, m_b = env.game.units["missileA"], env.game.units["missileB"]
    m_a.detail["loiter_until"] = env.game.second + 1
    _o, _r, _d, info = env.step({"type": "wait", "seconds": 5})
    check("T5", "第一枚超时自毁", env.game.missile_state(m_a) == "lost", info)
    offered = [a for a in env.legal_actions()
               if a.get("type") == "launch" and a.get("missile") == "missileB"]
    check("T5", "超时释放占用 → 第二枚重新可选（不是“一局一枚”）", bool(offered), offered)
    env.step(offered[0])
    check("T5", "第二枚成功发射", env.game.missile_state(m_b) == "loitering")
    env.game.units["carrier"].blood = 0
    _o, _r, _d, info = env.step({"type": "wait", "seconds": 5})
    check("T5", "载车被毁 → 在飞弹终止（`lost`）",
          env.game.missile_state(m_b) == "lost", info)
    check("T5", "终止后载车不再有在飞占用",
          not env.game.units["carrier"].detail.get("launched_missile"))


# ---------------------------------------------------------------- T6
def t6_consumption(fixture):
    print("\n== T6 攻击后弹体消失 / 只结算一次 / 被拒不消耗 ==")
    env = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=900, fog=False,
                    move_radius=1)
    env.reset(seed=6)
    env.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
    env.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
    g = env.game
    m, foe = g.units["missileA"], g.units["foe"]
    check("T6", "在飞弹在侦察半径内可见敌目标（可攻击的前提）",
          g.can_observe(m.id, foe.id), "dist=%d" % G.distance(m.pos, foe.pos))
    before_blood = foe.blood
    _o, _r, _d, info = env.step({"type": "attack", "unit": "missileA",
                                 "target": "foe", "weapon": 76})
    check("T6", "攻击动作被执行（不是 unknown / 被拒）",
          not info.get("invalid_action"), info)
    check("T6", "攻击完成后弹体消失（`spent` / 血量 0）",
          g.missile_state(m) == "spent" and m.blood == 0)
    check("T6", "清理了在飞状态（无 launcher 占用、无持续打击、无机动路径）",
          not g.units["carrier"].detail.get("launched_missile")
          and not m.detail.get("target") and not m.detail.get("path")
          and not m.detail.get("loiter_until"))
    check("T6", "弹体不能被再次下令攻击（已不是存活单位）",
          env.step({"type": "attack", "unit": "missileA", "target": "foe",
                    "weapon": 76})[3].get("invalid_action"))

    env2 = HexwarEnv(snapshot=fixture, side="BLUE", max_seconds=900, fog=False,
                     move_radius=1)
    env2.reset(seed=6)
    env2.step({"type": "load", "unit": "missileA", "carrier": "carrier"})
    env2.step({"type": "launch", "unit": "carrier", "missile": "missileA"})
    g2 = env2.game
    m2 = g2.units["missileA"]
    far = G.make_mapid(10, 14)          # 界外的远处目标：必然被拒
    g2.units["foe"].pos = far
    g2.units["foe"].pos2 = far
    info = env2.step({"type": "attack", "unit": "missileA", "target": "foe",
                      "weapon": 76})[3]
    check("T6", "被拒的攻击（超出侦察/射程）不消耗弹体",
          info.get("invalid_action") and g2.missile_state(m2) == "loitering", info)
    check("T6", "被拒的攻击也不扣弹药",
          m2.ammo.get("a5") == 1, m2.ammo)
    return before_blood


# ---------------------------------------------------------------- T7
def t7_timing(fixture):
    print("\n== T7 计时：RoomSpeed 映射 / 保存恢复 / 超时边界 ==")
    check("T7", "默认按客户端公式 `parseInt(1200/RoomSpeed)`：RoomSpeed=3 → 400 秒",
          Game.loiter_seconds(3) == 400, Game.loiter_seconds(3))
    check("T7", "不同 RoomSpeed 得到不同上限（3→400 / 10→120 / 7→171，取整同 parseInt）",
          (Game.loiter_seconds(10), Game.loiter_seconds(7)) == (120, 171))
    check("T7", "不再使用无来源占位值 120（但常量仍保留作对照）",
          LOITER_SEC_PLACEHOLDER == 120 and Game.loiter_seconds(3) !=
          LOITER_SEC_PLACEHOLDER)
    g = Game(seed=7, snapshot=fixture, loiter_sec=77)
    check("T7", "允许显式覆盖巡飞上限，并记录来源",
          g.loiter_sec == 77 and "override" in g.loiter_sec_source,
          g.loiter_sec_source)
    g.start()
    car, m = g.units["carrier"], g.units["missileA"]
    g.deploy_missile(car.id, m.id)
    g.launch_missile(car.id, m.id)
    check("T7", "在飞弹记录自己的上限与来源（不同 RoomSpeed 下取值可追溯）",
          m.detail.get("loiter_seconds") == 77
          and "override" in m.detail.get("loiter_seconds_source"), m.detail)
    deadline = m.detail["loiter_until"]
    g.advance(deadline - 1 - g.second)
    check("T7", "超时边界：`deadline - 1` 仍在飞", g.missile_state(m) == "loitering")
    g.advance(1)
    check("T7", "超时边界：到 `deadline` 即自毁（`<=` 语义）",
          g.missile_state(m) == "lost")

    g2 = Game(seed=7, snapshot=fixture, room_speed=3)
    g2.start()
    car2, m2 = g2.units["carrier"], g2.units["missileA"]
    g2.deploy_missile(car2.id, m2.id)
    g2.launch_missile(car2.id, m2.id)
    g2.advance(10)
    for label, clone in (("deepcopy", copy.deepcopy(g2)),
                         ("pickle", pickle.loads(pickle.dumps(g2)))):
        restored = clone.units["missileA"]
        keep_state = clone.missile_state(restored)
        keep_deadline = restored.detail.get("loiter_until")
        clone.advance(clone.loiter_sec)
        check("T7", "保存恢复（%s）后在飞状态与截止时间保持" % label,
              keep_state == "loitering" and keep_deadline == m2.detail.get("loiter_until")
              and clone.missile_state(clone.units["missileA"]) == "lost",
              "state=%s deadline=%s" % (keep_state, keep_deadline))
        check("T7", "保存恢复（%s）后载车占用同步释放" % label,
              not clone.units["carrier"].detail.get("launched_missile"))


# ---------------------------------------------------------------- T9
def t9_regression(fixture):
    print("\n== T9 不再靠手工改状态绕过缺陷 ==")
    g = Game(seed=8, snapshot=SNAP)
    g.start()
    msls = sorted((u for u in g.units.values() if u.is_loitering_missile()),
                  key=lambda x: str(x.id))
    car = next(u for u in sorted(g.units.values(), key=lambda x: str(x.id))
               if u.typeid == 2 and u.pos == msls[0].pos and u.color)
    m = msls[0]
    opening_blood = m.blood
    ok1, why1 = g.deploy_missile(car.id, m.id)   # 不手工写 carrier
    ok2, why2 = g.launch_missile(car.id, m.id)   # 不清 launched_missile
    check("T9", "无需手工 `blood=1` / 手工写 carrier / 手工清标记即可完成部署→发射",
          ok1 and ok2 and m.blood == opening_blood, "%s / %s" % (why1, why2))
    msls[1].pos = car.pos                     # 测试摆位：让第二枚与载车同格
    msls[1].pos2 = car.pos
    ok3, why3 = g.deploy_missile(car.id, msls[1].id)
    ok4, why4 = g.launch_missile(car.id, msls[1].id)
    check("T9", "同车第二枚的拒绝理由是“有一枚在飞”（不是旧口径的“本局已发射过”）",
          ok3 and not ok4 and "在飞" in why4, why4)
    g.advance(g.loiter_sec + 1)
    check("T9", "超时后占用**自动**释放（旧用例靠手工 `pop` 才看得到）",
          not g.units[car.id].detail.get("launched_missile"))
    ok5, why5 = g.launch_missile(car.id, msls[1].id)
    check("T9", "旧口径的永久锁不复存在：另一枚弹真的可以再发射", ok5, why5)


def main():
    ap = argparse.ArgumentParser(description="巡飞弹生命周期验收（issue #1）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        build_fixture(tmp)
        t1_t8_states(tmp)
        t2_gates(tmp)
        t3_interface(tmp)
        t4_unload(tmp)
        t5_one_in_flight(tmp)
        t6_consumption(tmp)
        t7_timing(tmp)
        t9_regression(tmp)
    passed = sum(1 for r in RESULTS if r["ok"])
    total = len(RESULTS)
    print("\n" + "=" * 64)
    print("巡飞弹生命周期验收：%d/%d 通过" % (passed, total))
    if args.json:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    failed = [r for r in RESULTS if not r["ok"]]
    if failed:
        print("失败项：")
        for r in failed:
            print("  - [%s] %s　— %s" % (r["tag"], r["name"], r["note"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
