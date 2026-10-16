# -*- coding: utf-8 -*-
"""新增规则实现的验证：手册 4.1.1(6) / 4.1.2(4)(5) / 4.8(2) / 4.9 / 4.10。

运行：``py -3.13 tools/verify_rules_extra.py``
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import Game                       # noqa: E402
from engine import geometry as geom           # noqa: E402

PASS = []
FAIL = []


def check(name, ok, note=""):
    (PASS if ok else FAIL).append(name)
    print(("  [通过] " if ok else "  [失败] ") + name + (("　— " + note) if note else ""))


g = Game(seed=1)
hexes = g.hexes
road_ids = [m for m, h in hexes.items() if (h.get("note") or "") in g.ROAD_NOTES]
step0 = [m for m, h in hexes.items() if int(h.get("step") or 0) == 0]
stepm1 = [m for m, h in hexes.items() if int(h.get("step") or 0) == -1]


def veh():
    for u in g.units.values():
        if u.typeid == 2 and u.alive() and not u.detail.get("carrier"):
            return u
    return None


def air():
    for u in g.units.values():
        if u.typeid == 3:
            return u
    return None


print("\n== 4.1.2(4) 道路：沿道路不受地形与高差影响 ==")
print("  （想定道路标记 %d 个，取自格 note ∈ %s）" % (len(road_ids), list(g.ROAD_NOTES)))
v = veh()
if v and road_ids:
    r = next((m for m in road_ids if geom.distance(m, v.pos) >= 0), road_ids[0])
    nb = sorted(geom.neighbors(r))
    frm = next((n for n in nb if n in hexes and int(hexes[n].get("step") or 0) != 0), None)
    if frm:
        sec_road, why_road = g.step_seconds(v, frm, r)
        check("车辆进入道路格可通行", sec_road > 0, "%.0fs %s" % (sec_road, why_road))
        # 道路格：不吃地形/坡度惩罚 → 应等于基础 12 秒（1 格车辆基准）
        check("道路格耗时 = 车辆基准（不受地形/高差影响）", sec_road == 12.0,
              "实际 %.0fs（居民地/丛林/坡度会放大）" % sec_road)
        # 对照：找一个居民地(cond=7)非道路格，应 ×3
        town = [m for m, h in hexes.items()
                if int(h.get("cond") or 0) == 7 and (h.get("note") or "") not in g.ROAD_NOTES]
        if town:
            t = town[0]
            tf = next((n for n in sorted(geom.neighbors(t))
                       if n in hexes and int(hexes[n].get("step") or 0) != 0), None)
            if tf:
                sec_town, _ = g.step_seconds(v, tf, t)
                check("对照：居民地格耗时 > 道路格", sec_town > sec_road,
                      "居民地 %.0fs vs 道路 %.0fs" % (sec_town, sec_road))
else:
    check("道路用例可用", False, "无车辆或无道路格，跳过")

print("\n== 4.1.2(5) 通行能力 < 通行等级 不可进入 ==")
print("  （想定 7527 实际 step∈{0} 的格 %d 个、step=-1 的格 %d 个 → 用合成格验证）"
      % (len(step0), len(stepm1)))
if v:
    nb = [n for n in sorted(geom.neighbors(v.pos)) if n in hexes]
    probe = next((n for n in nb
                  if not any(x.alive() and x.pos == n for x in g.units.values())), None)
    if probe:
        saved = dict(hexes[probe])
        hexes[probe]["step"] = 0
        sec0, why0 = g.step_seconds(v, v.pos, probe)
        check("step=0（禁行区）车辆不可进入", sec0 == 0, why0)
        hexes[probe]["step"] = -1
        old_c0 = v.c0
        v.c0 = 0
        sec_a, why_a = g.step_seconds(v, v.pos, probe)
        v.c0 = 2
        sec_b, _ = g.step_seconds(v, v.pos, probe)
        v.c0 = old_c0
        check("step=-1 且 c0≠2 → 车辆禁行", sec_a == 0, why_a)
        check("step=-1 且 c0=2 → 可通行", sec_b > 0, "%.0fs" % sec_b)
        hexes[probe].clear()
        hexes[probe].update(saved)
    else:
        check("找到空闲邻格", False, "邻格均被占，跳过")

print("\n== 4.8(2) 无人机滞空高度接入通视 ==")
a = air()
if a:
    e = g.elevation(a.pos)
    check("unit_height(空中单位) = 格高程 + 滞空高度",
          g.unit_height(a) == e + a.alt, "%d = %d + %d" % (g.unit_height(a), e, a.alt))
else:
    check("存在空中单位", False, "无 typeid=3")
if v:
    check("unit_height(地面单位) = 格高程", g.unit_height(v) == g.elevation(v.pos))

print("\n== 4.1.1(6) 机动中遭压制 → 解压制后续行 ==")
u = None
for cand in g.units.values():
    if cand.typeid == 1 and cand.alive():
        nbs = [n for n in sorted(geom.neighbors(cand.pos))
               if g.step_seconds(cand, cand.pos, n)[0] > 0]
        if len(nbs) >= 1:
            u = cand
            path = [nbs[0]]
            break
if u:
    ok, _ = g.begin_move(u.id, path)
    check("登记机动路径", ok and g.pending_moves() and u.id in g.pending_moves())
    u.stunned_until = g.second + 10
    u.keep = 1
    g.advance(5)
    check("压制中不推进（未完成当前格前停止）", u.pos != path[0], "pos=%d" % u.pos)
    g.advance(10)                        # 压制解除
    check("解压制后自动续行（手册 4.1.1(6)）", u.pos == path[0],
          "pos=%d 期望=%d" % (u.pos, path[0]))
    check("路径执行完毕后清空", u.id not in g.pending_moves())

print("\n== 4.9 无人战车：失去依托 → 失控 ==")
ugv = next((x for x in g.units.values() if x.is_ugv()), None)
cv = veh()
if ugv and cv:
    ugv.detail["carrier"] = cv.id
    ugv.detail.pop("out_of_control", None)
    cv.blood = 1
    g.advance(1)
    check("依托存在时不失控", not ugv.out_of_control())
    cv.blood = 0
    g.advance(1)
    check("依托被摧毁 → 失控（手册 4.9(3)）", ugv.out_of_control())
    tgt = next((x for x in g.units.values()
                if x.color != ugv.color and x.alive()), None)
    if tgt:
        ugv.blood = 1                       # 保证走到"失控"校验而不是"已被消灭"
        _r, err = g.attack(ugv.id, tgt.id)
        check("失控后不能射击", err is not None and "4.9(3)" in (err or ""), str(err))
else:
    check("存在无人战车与车辆", False, "无 UGV/车辆")

print("\n== 4.10 巡飞弹：生命周期（部署 / 在飞占用 / 消耗 / 超时 / 载车被毁） ==")
# 用**全新**的一局：`deploy_missile` 只接受「对抗开始前」（`second == 0`，手册 4.10(3)），
# 而上面的用例已经把 `g` 的时钟推进过了。
gm = Game(seed=7)
gm.start()
msls = sorted((x for x in gm.units.values() if x.is_loitering_missile()),
              key=lambda x: str(x.id))
car = next((x for x in sorted(gm.units.values(), key=lambda x: str(x.id))
            if x.typeid == 2 and x.pos == msls[0].pos and x.color), None)
if len(msls) == 3 and car is not None:
    check("想定初态的巡飞弹与载车同格、但**未建立所属关系**（`unassigned`）",
          all(gm.missile_state(m) == "unassigned" and not m.detail.get("carrier")
              and not m.detail.get("launcher") for m in msls))
    check("初态不凭同格/武器 76 静默推断已发射（无 `launched_missile` 占用）",
          all(not x.detail.get("launched_missile") for x in gm.units.values()))
    ok, why = gm.launch_missile(car.id, msls[0].id)
    check("未部署上车不能发射（手册 4.10(3)）", not ok and "4.10(3)" in why, why)
    ok, why = gm.deploy_missile(car.id, msls[1].id)
    check("不同格的弹不能装到这辆车上（手册 3.3）", not ok and "同格" in why, why)
    ok, why = gm.deploy_missile(car.id, msls[0].id)
    check("对抗开始前可部署上车（手册 4.10(3)）", ok, why)
    check("部署后显式状态 = `carried`，且车内读不到兵力/阵营",
          gm.missile_state(msls[0]) == "carried" and msls[0].blood == 0
          and not msls[0].color, "blood=%s color=%r" % (msls[0].blood, msls[0].color))
    old_second = gm.second
    gm.second = 5
    ok, why = gm.deploy_missile(car.id, msls[1].id)
    check("对抗开始**之后**不能再上车（手册 4.10(3)）", not ok, why)
    gm.second = old_second
    gm.started = False
    ok, why = gm.launch_missile(car.id, msls[0].id)
    check("对抗开始前不能发射（手册 4.10(3)）", not ok and "开始前" in why, why)
    gm.started = True
    ok, why = gm.launch_missile(car.id, msls[0].id)
    check("车内 blood=0 也能发射（不再被 alive 门拦）", ok, why)
    check("发射后恢复阵营与编制兵力、状态 = `loitering`",
          msls[0].color == car.color and msls[0].blood > 0
          and gm.missile_state(msls[0]) == "loitering")
    check("发射后弹落在载车所在格", msls[0].pos == car.pos)
    check("发射把弹从载车 carrying 移出（不能重复发射/重复卸载）",
          msls[0].id not in (car.detail.get("carried") or []))
    check("载车记录在飞占用（`launched_missile` + `stop=1`）",
          car.detail.get("launched_missile") == msls[0].id and car.detail.get("stop") == 1)
    check("发射时记录巡飞上限与取值来源",
          msls[0].detail.get("loiter_seconds") == gm.loiter_sec
          and bool(msls[0].detail.get("loiter_seconds_source")))
    ok, why = gm.launch_missile(car.id, msls[0].id)
    check("同一辆车不能对同**一枚**弹重复发射", not ok, why)
    # 第二枚：先摆到同一格再上车（这一步只是测试摆位，不代表规则允许跳格装载）
    msls[1].pos = car.pos
    msls[1].pos2 = car.pos
    ok, why = gm.deploy_missile(car.id, msls[1].id)
    check("同格的第二枚弹可以上车", ok, why)
    ok, why = gm.launch_missile(car.id, msls[1].id)
    check("同车已有在飞弹 → 拒绝第二枚（手册 4.10(4)）", not ok and "在飞" in why, why)
    # 第一枚超时后释放占用 —— 且**不是**“一局一枚”的永久锁
    gm.advance(gm.loiter_sec + 1)
    check("巡飞超时自毁（手册 4.10(5)）", not msls[0].alive()
          and gm.missile_state(msls[0]) == "lost", "血=%d" % msls[0].blood)
    check("超时释放载车占用（原实现是永久锁：第二枚永远发不出去）",
          not car.detail.get("launched_missile") and car.detail.get("stop") == 0)
    ok, why = gm.launch_missile(car.id, msls[1].id)
    check("释放后可发射另一枚合法可用弹（手册 4.10(4) 是“同时一枚”）", ok, why)
    car.blood = 0
    gm.advance(1)
    check("发射车被摧毁 → 在飞弹自毁（手册 4.10(5)）", not msls[1].alive()
          and gm.missile_state(msls[1]) == "lost")
    check("已损失的弹不能复活（再次发射被拒）",
          not gm.launch_missile(car.id, msls[1].id)[0])
else:
    check("存在 3 枚巡飞弹与同格载车", False, "巡飞弹 %d 枚 / 载车 %s" % (len(msls), car))

print("\n" + "=" * 60)
print("新增规则验证：%d/%d 通过" % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
