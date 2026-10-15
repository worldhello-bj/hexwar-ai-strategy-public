"""逐条验证按手册新编码的规则是否真的生效。

每条检查都对应手册的具体条款号，失败会明确报出来。
运行：``py -3.13 tools/verify_rules.py``
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import crt                                   # noqa: E402
from engine import paths as _paths                       # noqa: E402
from engine.game import Game, SUPPRESS_SEC               # noqa: E402
from engine.equip import WEAPONS                         # noqa: E402
from env.hexwar_env import HexwarEnv                     # noqa: E402
from engine.geometry import distance, neighbors          # noqa: E402

# 反装甲武器（用 wz[0] 可能拿到高射机枪之类的非反装甲武器，会得出 0 战果）
ANTI_ARMOR = {"中型坦克炮", "重型坦克炮", "速射炮", "车载导弹", "便携导弹"}


def best_gun(u):
    """挑一把对装甲有效的武器；没有就退回首把。"""
    for w in u.wz:
        info = WEAPONS.get(w)
        if info and info[0] in ANTI_ARMOR:
            return w
    return u.wz[0] if u.wz else 0

SNAP = os.environ.get("HEXWAR_SNAP") or _paths.snapshot_dir()   # 默认=比赛初设（想定 7527）

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, note: str = "") -> None:
    RESULTS.append((name, bool(cond), note))
    print(("  [通过] " if cond else "  [失败] ") + name + (("　— " + note) if note else ""))


def new_game(seed=1):
    g = Game(seed=seed, snapshot=SNAP)
    g.start()
    g.second = 0
    return g


def pick(g, **want):
    """按属性找一个单位：pick(g, typeid=2, color='BLUE')。"""
    for u in g.units.values():
        if all(getattr(u, k, None) == v for k, v in want.items()):
            return u
    return None


def put(u, hexid):
    u.pos = hexid
    u.pos2 = hexid


# ----------------------------------------------------------------------
print("\n== 4.6.2(4) 直瞄对车辆产生有效战果 → 同时压制 ==")
g = new_game()
a = pick(g, typeid=2, color="BLUE")
t = pick(g, typeid=2, color="RED")
put(a, 80020)
put(t, 80019)
r = crt.direct_fire(shooter=dict(typeid=2, units=1, moving=False, stunned=False),
                    target=dict(typeid=2, armor="复合装甲", hidden=False,
                                moving=False, stacked=False),
                    weapon_name="重型坦克炮", dist=1, roll=2, fix_roll=8)
check("车辆 kills>0 时 suppressed=True",
      r["kills"] > 0 and r["suppressed"], "kills=%s suppressed=%s" % (r["kills"], r["suppressed"]))

print("\n== 4.6.2(5) 战果被防护力修正为 0 → 仅压制 ==")
found = None
for armor in ("无装甲", "中型装甲", "复合装甲", "重型装甲"):
    for wn in ("车载机枪", "速射炮", "中型坦克炮", "重型坦克炮"):
        for dist in (1, 4, 8, 10):
            for roll in range(2, 13):
                rr = crt.direct_fire(
                    shooter=dict(typeid=2, units=1, moving=False, stunned=False),
                    target=dict(typeid=2, armor=armor, hidden=True,
                                moving=True, stacked=False),
                    weapon_name=wn, dist=dist, roll=roll, fix_roll=roll)
                if rr.get("kills_raw", 0) > 0 and rr.get("final") == 0:
                    found = (armor, wn, dist, roll, rr)
                    break
            if found:
                break
        if found:
            break
    if found:
        break
if found:
    check("修正前战果>0、修正后恰为 0 时 → 压制（4.6.2(5)）",
          found[4]["suppressed"] and found[4]["kills"] == 0,
          "%s/%s/dist=%d/roll=%d  kills_raw=%d final=%d"
          % (found[0], found[1], found[2], found[3], found[4]["kills_raw"],
             found[4]["final"]))
else:
    check("修正前战果>0、修正后恰为 0 时 → 压制（4.6.2(5)）", False, "未能构造出该组合")

print("\n== 4.6.2(6) 修正后战果为负 → 无效（不压制）==")
found6 = None
for armor in ("无装甲", "中型装甲", "复合装甲", "重型装甲"):
    for wn in ("车载机枪", "速射炮", "中型坦克炮", "重型坦克炮"):
        for dist in (1, 4, 8, 10):
            for roll in range(2, 13):
                rr = crt.direct_fire(
                    shooter=dict(typeid=2, units=1, moving=False, stunned=False),
                    target=dict(typeid=2, armor=armor, hidden=True,
                                moving=True, stacked=False),
                    weapon_name=wn, dist=dist, roll=roll, fix_roll=roll)
                if rr.get("kills_raw", 0) > 0 and rr.get("final", 0) < 0:
                    found6 = (armor, wn, dist, roll, rr)
                    break
            if found6:
                break
        if found6:
            break
    if found6:
        break
if found6:
    check("修正后战果为负 → 无效且不压制（4.6.2(6)）",
          found6[4]["result"] == crt.INVALID and found6[4]["kills"] == 0
          and not found6[4]["suppressed"],
          "%s/%s/dist=%d/roll=%d  kills_raw=%d final=%d → %r"
          % (found6[0], found6[1], found6[2], found6[3], found6[4]["kills_raw"],
             found6[4]["final"], found6[4]["result"]))
else:
    check("修正后战果为负 → 无效且不压制（4.6.2(6)）", False, "未能构造出该组合")

print("\n== 4.6.1 人员战果：消灭与压制互斥 ==")
r = crt.direct_fire(shooter=dict(typeid=1, units=3, moving=False, stunned=False),
                    target=dict(typeid=1, armor="无装甲", hidden=False,
                                moving=False, stacked=True),
                    weapon_name="步兵轻武器", dist=1, roll=2, fix_roll=8)
check("人员被消灭时不附带压制", (r["kills"] == 0 or not r["suppressed"]),
      "kills=%s suppressed=%s" % (r["kills"], r["suppressed"]))

# ----------------------------------------------------------------------
print("\n== 4.6.3(2) 已压制人员再被裁决压制/受伤 → 额外损失 1 个班 ==")
g = new_game()
a = pick(g, typeid=2, color="RED")
t = pick(g, typeid=1, color="BLUE")          # 用蓝方人员当靶子（裁决逻辑与颜色无关）
if a is None or t is None:
    check("再压制额外扣兵", False, "样例缺少红方车辆或蓝方人员")
else:
    # 需要构造"本次裁决为压制且没有新伤亡"的情形；先把两者放到相邻格，
    # 否则通视/观察校验会直接拒掉每一次射击。
    put(a, 80020)
    put(t, 80019)
    got = None
    for _ in range(300):
        t.blood = 4
        t.stunned_until = g.second + SUPPRESS_SEC
        before = t.blood
        rr, err = g.attack(a.id, t.id, dist=1)
        if err is None and rr["suppressed"] and rr["kills"] == 0:
            got = before - t.blood
            break
    check("已压制人员再次被裁决压制 → 额外损失 1 个班",
          got == 1, "本次扣减 %s 个班" % got)

print("\n== 4.6.3(1) 人员被压制不能机动 ==")
g = new_game()
p = pick(g, typeid=1, color="BLUE")
p.stunned_until = g.second + SUPPRESS_SEC
nb = neighbors(p.pos)[0]
sec, why = g.step_seconds(p, p.pos, nb)
check("被压制的人员无法机动", sec <= 0, "返回原因：%s" % why)

print("\n== 4.1.1(4) 冲锋里程疲劳 + 疲劳不能冲锋 ==")
g = new_game()
p = pick(g, typeid=1, color="BLUE")
p.ste = 4
nbs = [n for n in neighbors(p.pos) if g.hex(n)]
sec, _ = g.step_seconds(p, p.pos, nbs[0])
if sec > 0:
    g.commit_step(p, p.pos, nbs[0])
charged = p.fatigued(g.second)
sec2, why2 = g.step_seconds(p, p.pos, nbs[1])
check("冲锋 200 米后进入疲劳", charged, "疲劳标记=%s" % charged)
check("疲劳状态下不能继续冲锋", sec2 <= 0, "返回原因：%s" % why2)

print("\n== 4.1.2(3) 车辆高差 >100 米不可通过 ==")
g = new_game()
v = pick(g, typeid=2, color="BLUE")
base = g.elevation(v.pos)
cand = next((n for n in neighbors(v.pos) if g.hex(n)), None)
if cand is None:
    check("高差>100 米车辆被拒", False, "找不到相邻格")
else:
    # 样例地图高差不够，临时抬高一格以构造条件
    hb = g.hexes[cand]
    old = hb.get("elevation")
    hb["elevation"] = base + 101
    sec, why = g.step_seconds(v, v.pos, cand)
    hb["elevation"] = old
    check("高差>100 米车辆被拒", sec <= 0, "高差=101，原因=%s" % why)

# ----------------------------------------------------------------------
print("\n== 4.8(1) 无人机看掩蔽车辆不减半、看掩蔽人员减半 ==")
g = new_game()
drone = None
for u in g.units.values():
    if u.typeid == 3:
        drone = u
        break
if drone is None:
    check("无人机侦察距离规则", False, "样例中没有空中单位")
else:
    veh = pick(g, typeid=2, color="RED")
    per = pick(g, typeid=1, color="BLUE")      # 侦察规则与颜色无关
    if veh is None or per is None:
        check("无人机侦察距离规则", False, "缺少车辆或人员单位")
    else:
        r0 = g.observe_range(drone, veh)
        veh.ste = 2
        r1 = g.observe_range(drone, veh)
        p0 = g.observe_range(drone, per)
        per.ste = 2
        p1 = g.observe_range(drone, per)
        check("掩蔽中的车辆不被减半", abs(r1 - r0) < 1e-9, "%s → %s" % (r0, r1))
        check("掩蔽中的人员被减半", p1 < p0, "%s → %s" % (p0, p1))

print("\n== 4.7 手雷：全格（含己方）裁决；不打空中/掩体 ==")
g = new_game()
thrower = None
for u in g.units.values():
    if u.typeid == 1 and 144 in u.wz:
        thrower = u
        break
if thrower is None:
    check("手雷规则", False, "样例中没有带手雷的人员单位")
else:
    target = neighbors(thrower.pos)[0]
    friend = pick(g, typeid=1, color=thrower.color)
    if friend is not None and friend is not thrower:
        put(friend, target)
    air = None
    for u in g.units.values():
        if u.typeid == 3 and u is not thrower:
            air = u
            break
    if air is not None:
        air_blood0 = air.blood
        put(air, target)
    res, err = g.grenade(thrower.id, target)
    check("手雷对格内目标逐一裁决", err is None and len(res or []) >= 1,
          "裁决 %d 个目标" % len(res or []))
    if air is not None:
        note = next((m for u, m in (res or []) if u is air), "（未出现在裁决结果中）")
        check("手雷不伤空中单位（手册 4.7(2)）",
              air.blood == air_blood0 and "不能" in note, note)
    if friend is not None and friend is not thrower:
        check("手雷对己方同格单位也裁决（手册 4.7(1)）", friend.id in
              {u.id for u, _ in (res or [])}, "己方单位在格内")

print("\n== 4.3(2) 空中单位不能夺控 ==")
g = new_game()
air = next((u for u in g.units.values() if u.typeid == 3), None)
if air is None or not g.cities:
    check("空中单位不能夺控", False, "样例缺少空中单位或夺控点")
else:
    put(air, g.cities[0]["pos"])
    for u in g.units.values():
        if u.color != air.color and u.typeid in (1, 2):
            put(u, 80019)            # 挪远，避免干扰
    check("空中单位不能夺控", g.apply_city_capture(air) is None)

# ----------------------------------------------------------------------
print("\n== 3.3 上车 / 下车 ==")
env = HexwarEnv(snapshot=SNAP, side="BLUE", fog=False)
env.reset(seed=2)
g = env.game
inf = next((u for u in g.units.values() if u.typeid == 1 and u.color == "BLUE"), None)
veh = next((u for u in g.units.values() if u.typeid == 2 and u.color == "BLUE"), None)
if inf is None or veh is None:
    check("上车/下车", False, "样例缺少 BLUE 人员或车辆")
else:
    put(veh, 80020)
    put(inf, 80019)
    _o, _r, _d, info = env._apply({"type": "load", "unit": inf.id, "carrier": veh.id})
    check("人与车不同格时不能上车（手册 3.3）",
          inf.detail.get("carrier") is None, info.get("msg"))

    put(inf, veh.pos)
    _o, _r, _d, info = env._apply({"type": "load", "unit": inf.id, "carrier": veh.id})
    check("同格可以上车", inf.detail.get("carrier") == veh.id, info.get("msg"))

    # 随车机动
    dest = [n for n in neighbors(veh.pos) if g.hex(n) and g.step_seconds(veh, veh.pos, n)[0] > 0][0]
    _o, _r, _d, _i = env._apply({"type": "move", "unit": veh.id, "path": [veh.pos, dest]})
    check("已上车步兵随车机动", inf.pos == veh.pos, "步兵 %s / 车 %s" % (inf.pos, veh.pos))

    # 已上车不再独立行动
    ids = {a["unit"] for a in env.legal_actions() if a["type"] == "move"}
    check("已上车的步兵不再产生独立机动动作", inf.id not in ids)

    # 下车
    _o, _r, _d, info = env._apply({"type": "unload", "unit": veh.id})
    check("下车后解除搭载关系", inf.detail.get("carrier") is None, info.get("msg"))

    # 车辆被压制时不能下车
    put(inf, veh.pos)
    env._apply({"type": "load", "unit": inf.id, "carrier": veh.id})
    veh.stunned_until = g.second + SUPPRESS_SEC
    _o, _r, _d, info = env._apply({"type": "unload", "unit": veh.id})
    check("车辆被压制时不能下车（手册 4.6.3(1)(4)）",
          inf.detail.get("carrier") == veh.id, info.get("msg"))

print("\n== 4.5(2) 打击间隔期间不允许掩蔽 ==")
env = HexwarEnv(snapshot=SNAP, side="BLUE", fog=False)
env.reset(seed=3)
g = env.game
sh = next((u for u in g.units.values() if u.typeid == 2 and u.color == "BLUE"), None)
tg = next((u for u in g.units.values() if u.color == "RED"), None)
if sh is None or tg is None:
    check("打击间隔禁止掩蔽", False, "样例缺少单位")
else:
    put(sh, 80020)
    put(tg, 80019)
    env._apply({"type": "attack", "unit": sh.id, "target": tg.id, "weapon": sh.wz[0]})
    _o, _r, _d, info = env._apply({"type": "hide", "unit": sh.id, "on": 1})
    check("刚射击过的棋子不能立刻掩蔽", sh.ste != 2, info.get("msg"))
    acts = [a for a in env.legal_actions() if a["type"] == "hide" and a["unit"] == sh.id]
    check("legal_actions 也不提供该掩蔽动作", not acts)

print("\n== 4.11 同格交战自动触发 ==")
g = new_game()
a = pick(g, typeid=2, color="BLUE")
t = pick(g, typeid=2, color="RED")
if a is None or t is None:
    check("同格交战", False, "样例缺少红蓝车辆")
else:
    put(a, 80020)
    put(t, 80020)                 # 强制同格
    before = t.blood + a.blood
    rounds = g.same_hex_engage()
    check("同格触发自动交战裁决", len(rounds) >= 1, "裁决 %d 次" % len(rounds))
    check("交战间隔为正常间隔的 1/3（手册 4.11(3)）",
          g.ENGAGE_INTERVAL_SEC == max(1, g.ATTACK_INTERVAL_SEC // 3),
          "间隔=%ds" % g.ENGAGE_INTERVAL_SEC)

# ----------------------------------------------------------------------
print("\n== 3.14 设定目标后自动持续打击 ==")
env = HexwarEnv(snapshot=SNAP, side="BLUE", fog=False, max_seconds=1800)
env.reset(seed=5)
g = env.game
sh = pick(g, typeid=2, color="BLUE")
tg = pick(g, typeid=2, color="RED")
if sh is None or tg is None:
    check("持续打击", False, "样例缺少红蓝车辆")
else:
    nbs = [n for n in neighbors(80019) if g.hex(n)]
    put(tg, 80019)
    put(sh, nbs[0])
    tg.blood = 4
    before = tg.blood
    _o, _r, _d, info = env.step({"type": "attack", "unit": sh.id,
                                 "target": tg.id, "weapon": best_gun(sh), "track": 1})
    check("设定持续打击目标成功（手册 3.14）",
          sh.detail.get("target") == tg.id, info.get("msg"))

    # 之后**只等待、不再下任何命令**
    for _ in range(30):
        env.step({"type": "wait", "seconds": 10})
        if tg.blood <= 0:
            break
    check("无需重复下令即持续造成伤害", tg.blood < before,
          "血量 %d → %d" % (before, tg.blood))
    check("目标被消灭后自动解除持续打击",
          tg.blood <= 0 and not sh.detail.get("target"),
          "血量=%d 仍挂目标=%s" % (tg.blood, sh.detail.get("target")))

print("\n== 3.16 集火射击：对格坐标，目标离开即失败 ==")
env = HexwarEnv(snapshot=SNAP, side="BLUE", fog=False, max_seconds=1800)
env.reset(seed=6)
g = env.game
bs = [u for u in g.units.values() if u.typeid == 2 and u.color == "BLUE"][:2]
rt = pick(g, typeid=2, color="RED")
if not bs or rt is None:
    check("集火射击", False, "样例缺少足够单位")
else:
    nbs = [n for n in neighbors(80019) if g.hex(n)]
    put(bs[0], nbs[0])
    if len(bs) > 1 and len(nbs) > 1:
        put(bs[1], nbs[1])
    put(rt, 80019)
    rt.blood = 8
    b0 = rt.blood
    gun = best_gun(bs[0])
    res, err = g.area_fire([u.id for u in bs], 80019, weapon_id=gun)
    check("格内有目标时集火生效", err is None and len(res["shots"]) >= 1,
          "裁决 %d 次" % len((res or {}).get("shots") or []))
    check("集火造成伤害", rt.blood < b0, "血量 %d → %d" % (b0, rt.blood))

    far = next((h for h in g.hexes if distance(h, 80019) > 5), None)
    if far is None:
        check("目标离开后集火失败", False, "找不到足够远的格子")
    else:
        put(rt, far)
        b1 = rt.blood
        res2, err2 = g.area_fire([u.id for u in bs], 80019, weapon_id=gun)
        check("目标已离开该格 → 本次集火失败（手册 3.16，不跟踪）",
              err2 is None and res2["result"] == "MISS" and rt.blood == b1,
              "result=%s 血量 %d → %d" % ((res2 or {}).get("result"), b1, rt.blood))

# ----------------------------------------------------------------------
ok = sum(1 for _n, c, _m in RESULTS if c)
print("\n" + "=" * 60)
print("规则验证：%d/%d 通过" % (ok, len(RESULTS)))
failed = [n for n, c, _m in RESULTS if not c]
if failed:
    print("未通过：")
    for n in failed:
        print("  - " + n)
sys.exit(0 if not failed else 1)
