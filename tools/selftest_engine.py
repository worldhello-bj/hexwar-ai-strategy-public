# -*- coding: utf-8 -*-
"""裁决引擎自检：几何、通视、机动、直瞄裁决、夺控、计分。"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Game, geometry as G, crt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "engine_selftest.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)   # 公开克隆里可能没有 analysis/
L = []
def w(x=""):
    L.append(str(x))

w("# 裁决引擎自检")
w()

g = Game(seed=20261014)
w("## 1. 数据装载")
w()
w("- 地图格: %d" % len(g.hexes))
w("- 单位: %d (红 %d / 蓝 %d / 未上阵 %d)" % (
    len(g.units),
    sum(1 for u in g.units.values() if u.color == "RED"),
    sum(1 for u in g.units.values() if u.color == "BLUE"),
    sum(1 for u in g.units.values() if not u.color)))
w("- 城市: %d" % len(g.cities))
w()

w("## 2. 几何校验（mapid ↔ 行/列）")
w()
ok = True
for mid in (80014, 80015, 80019, 80020, 170068, 120037, 250113):
    row, col = G.split_mapid(mid)
    back = G.make_mapid(row, col)
    hx = G.hexid(mid)
    mark = "✓" if back == mid else "✗"
    if back != mid:
        ok = False
    w("- mapid=%d → 行%d 列%d → 回代 %d %s   hexid=%s" % (mid, row, col, back, mark, hx))
w()
w("- 往返一致: **%s**" % ("是" if ok else "否"))
w()

w("## 3. 邻接与距离")
w()
a = 120037
nb = G.neighbors(a)
w("- %d 的 6 个邻格: %s" % (a, ["%d(距离%d)" % (n, G.distance(a, n)) for n in nb]))
w("- 邻格距离均为 1: **%s**" % ("是" if all(G.distance(a, n) == 1 for n in nb) else "否"))
same = [n for n in nb if G.hexid(n) == G.hexid(a)]
w("- 邻格 hexid 与自身不同: **%s**" % ("是" if not same else "否 %s" % same))
w()

w("## 4. 通视（手册 4.4）")
w()
pairs = [(120037, 120039), (120037, 130037), (120037, 160060), (120037, 90014)]
for x, y in pairs:
    d = G.distance(x, y)
    okv, blk = g.los(x, y)
    w("- %s(%d) → %s(%d) 距离%d 通视=%s 阻挡格=%s 高程差=%d" % (
        G.hexid(x), x, G.hexid(y), y, d, "是" if okv else "否",
        [G.hexid(b) for b in blk][:6], g.elevation(x) - g.elevation(y)))
w()

w("## 5. 机动耗时（手册 4.1）")
w()
tank = next((u for u in g.units.values() if u.name in ("坦克", "中型坦克") and u.alive()), None)
inf = next((u for u in g.units.values() if u.typeid == 1 and u.alive()), None)
if inf is None:
    inf = next((u for u in g.units.values() if u.typeid == 1), None)
    if inf is not None:
        inf.blood = 2
for u in (tank, inf):
    if u is None:
        continue
    w("- **%s** (%s, typeid=%d) 位于 %s" % (u.army or u.name, u.name, u.typeid, G.hexid(u.pos)))
    for n in G.neighbors(u.pos):
        sec, why = g.step_seconds(u, u.pos, n)
        h = g.hex(n)
        w("    - → %s 地形=%s 高程=%s 耗时=%.0fs %s" % (
            G.hexid(n), g.terrain_kind(n), h["elevation"] if h else "-", sec, ("[" + why + "]") if why else ""))
    w()

w("## 6. 直瞄裁决抽样")
w()
red = [u for u in g.units.values() if u.color == "RED" and u.alive()]
blue = [u for u in g.units.values() if u.color == "BLUE" and u.alive()]
w("- 红方 %d 个、蓝方 %d 个存活单位" % (len(red), len(blue)))
if red and blue:
    best = None
    for s0 in red:
        for t0 in blue:
            d0 = G.distance(s0.pos, t0.pos)
            if best is None or d0 < best[0]:
                best = (d0, s0, t0)
    d, s, t = best
    w("- 最近红蓝对：%s → %s，距离 %d 格（%s → %s）" % (
        s.name, t.name, d, G.hexid(s.pos), G.hexid(t.pos)))
    w("- 通视: %s" % ("是" if g.los(s.pos, t.pos)[0] else "否"))
    w()
    w("| 武器 | 随机数 | 攻击等级 | 结果 |")
    w("|---|---|---|---|")
    for wname in ("重型坦克炮", "中型坦克炮", "车载机枪", "便携导弹", "步兵轻武器", "高射机枪"):
        for roll in (2, 7, 12):
            r = crt.direct_fire(
                {"typeid": s.typeid, "units": max(1, s.blood), "name": s.name},
                {"typeid": t.typeid, "armor": t.armor, "name": t.name},
                wname, d, roll=roll)
            res = "不适用" if r["result"] in ("NO_WEAPON", "IMMUNE") else repr(r["result"])
            w("| %s | %d | %s | %s |" % (wname, roll, r.get("level"), res))
    w()
    # 实战走一遍 game.attack（含射程/通视/观察/弹药校验）
    w("### 经 game.attack 完整链路（含射程、通视、观察、弹药校验）")
    w()
    for wname in ("重型坦克炮", "便携导弹"):
        wid = next((k for k, v in crt.__dict__.get("_T_WEAPONS", {}).items()), None)
    import engine.equip as EQ
    for wid in s.wz:
        info = EQ.WEAPONS.get(wid)
        if not info:
            continue
        wname = info[0]
        r, err = g.attack(s.id, t.id, weapon_id=wid)
        w("- %s(id=%d, 射程%d) → %s" % (
            wname, wid, info[2],
            (err if err else "裁决执行：%s（%s）" % (r["result"], "；".join(r["detail"])))))
    w()
    w("- 目标当前兵力: %d（初始 2）" % t.blood)
    w("- 目标压制状态: %s" % ("是" if t.stunned(g.second) else "否"))
    w()
w("## 7. 反无人机裁决（表9/表10）")
w()
w("（两颗骰均取 6；攻级取自表9，表10 的行 = 随机数）")
for wname in ("高射机枪", "速射炮", "激光", "便携防空导弹"):
    row = []
    for d in (0, 1, 3, 5):
        r = crt.direct_fire({"typeid": 2, "units": 1}, {"typeid": 3, "armor": "无装甲"},
                            wname, d, roll=6, fix_roll=6)
        if r["result"] == "NO_WEAPON":
            row.append("d%d:不适用" % d)
        else:
            row.append("d%d:攻级%s→%s" % (d, r.get("level"),
                                        "歼灭" if r["kills"] else "无效"))
    w("- %-8s %s" % (wname, " ".join(row)))
w()

w("## 8. 夺控判定（手册 4.3）")
w()
c = g.cities[0]
w("- 夺控点: %s 位于 %s（当前旗=%s）" % (c.get("name"), G.hexid(c["pos"]), c.get("flag")))
red_u = next((u for u in g.units.values() if u.color == "RED" and u.alive()), None)
if red_u:
    zone = [c["pos"]] + G.neighbors(c["pos"])
    enemy = [u.id for u in g.units.values()
             if u.alive() and u.color == "BLUE" and u.pos in zone]
    w("- 若红方单位进入该格：区域内蓝方地面单位 %d 个 → %s" % (
        len(enemy), "可夺控" if not enemy else "不可夺控"))
    red_u.pos = c["pos"]
    got = g.apply_city_capture(red_u)
    w("- 实际执行结果: %s" % (("夺控成功，旗变为 %s" % got["flag"]) if got else "未夺控"))
w()

w("## 9. 战斗日志")
w()
for line in g.log[-10:]:
    w("- %s" % line)
w()

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("selftest written:", OUT)
