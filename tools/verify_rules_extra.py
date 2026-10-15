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

print("\n== 4.10 巡飞弹：部署 / 一车一枚 / 超时自毁 / 发射车被毁自毁 ==")
msl = next((x for x in g.units.values() if x.is_loitering_missile()), None)
cv2 = veh()
if msl and cv2:
    msl.blood = 1
    msl.detail["carrier"] = cv2.id
    cv2.blood = 1
    ok, why = g.launch_missile(cv2.id, msl.id)
    check("部署在该车上 → 可发射（手册 4.10(3)）", ok, why)
    check("发射后巡飞弹落在发射车所在格", msl.pos == cv2.pos)
    ok2, why2 = g.launch_missile(cv2.id, msl.id)
    check("一车一次只能发射一枚（手册 4.10(4)）", not ok2, why2)
    # 超时自毁
    msl.blood = 1
    g.advance(g.LOITER_SEC + 1)
    check("巡飞超时自毁（手册 4.10(5)）", not msl.alive(), "血量=%d" % msl.blood)
    # 发射车被毁 → 自毁
    msl2 = next((x for x in g.units.values()
                 if x.is_loitering_missile() and x.id != msl.id), None)
    if msl2 and cv2:
        msl2.blood = 1
        msl2.detail["carrier"] = cv2.id
        cv2.detail.pop("launched_missile", None)
        g.launch_missile(cv2.id, msl2.id)
        cv2.blood = 0
        g.advance(1)
        check("发射车被摧毁 → 巡飞弹自毁（手册 4.10(5)）", not msl2.alive())
else:
    check("存在巡飞弹与车辆", False, "无巡飞弹/车辆")

print("\n" + "=" * 60)
print("新增规则验证：%d/%d 通过" % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
