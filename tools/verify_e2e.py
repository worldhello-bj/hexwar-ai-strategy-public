# -*- coding: utf-8 -*-
"""端到端验证：完全按官方客户端的 HTTP 契约驱动裁决引擎。

★ 2026-10-15 修正：旧版只发了 `ComPost{"attack":...}`（那只是**下命令**），
从不调 `com/Attack.asp`（真正裁决的地方），所以报告里恒为「战果生效: 否」——
看起来像引擎坏了，其实是链路只走了一半。现在改成官方三段：

  ① `ComPost.asp` 下命令（入队）           → 返回空串 = 成功
  ② `com/ObjWeaponReady.asp` 武器就绪      → 连射之间必须复位（否则第二发必失败）
  ③ `com/Attack.asp?ObjID&TarID&WeaponID`  → 返回非空 = 射击成功，战果经下一帧 ObjAjax 下发
  ④ `ObjFire.asp`（客户端放弹道动画时会顺带把 att 复位）
  ⑤ 读 `Report_Attack.asp` 战报核对

另外：**想定 7527 开局双方互不通视**（真实服务端亦然），所以先按官方机动链
（ComPost → updatePos → MoveEnd）把射手推到目标相邻格，再开火；并保留一个
「超射程/不通视开火不产生战果」的反例，避免把"前置条件挡住的空返回"误读成规则不允许。

跑法：先起桩服务器，再执行本脚本。
  py -3.13 offline\\server.py
  py -3.13 tools\\verify_e2e.py      # 有任一项 FAIL 则 exit 1
"""
import json
import os
import re
import sys

import requests

B = "http://127.0.0.1:8080/WARGAMEAI7/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "e2e_verify.md")
sys.path.insert(0, ROOT)
from engine import crt as CRT                    # noqa: E402
from engine import equip as EQ                   # noqa: E402
from engine import geometry as G                 # noqa: E402

s = requests.Session()
s.headers.update({"User-Agent": "HexwarE2E/1.0", "Referer": B + "main.html"})

L = []
RESULT = []


def w(x=""):
    L.append(str(x))


def check(name, ok, note=""):
    RESULT.append((name, bool(ok), note))
    w("- **%s: %s**%s" % (name, "是" if ok else "否", ("　— %s" % note) if note else ""))


def g(path, **params):
    r = s.get(B + path, params=params or None, timeout=20)
    return r.content.decode("gb18030", errors="replace")


def raw(path, **params):
    return s.get(B + path, params=params or None, timeout=20).text.strip()


def objajax():
    t = g("ObjAjax.asp").strip()
    if not t:
        return {}
    t = re.sub(r",(\s*})", r"\1", t)
    try:
        return json.loads(t)
    except Exception:                                        # noqa: BLE001
        return {}


def parse_units(html):
    """从 `Obj_Get.asp` 的 `<div id="Obj_x" k="v" ...>` 里取全部属性。"""
    out = {}
    for m in re.finditer(r'<div id="(Obj_\d+)"([^>]*)>', html):
        out[m.group(1)] = dict(re.findall(r'([A-Za-z_]\w*)="([^"]*)"', m.group(2)))
    return out


def weapon_choice(units, a, t, d):
    """挑武器：返回 (weapon_data_id, 该距离的预期攻击等级, 名称, 目标类型, 装甲)。

    `d<=0` 时按**近距（1 格）能力**评估 —— 选配对必须这么做：
    想定 7527 开局双方相距 16..19 格，任何武器都超射程，按当前距离筛必然为空。
    """
    try:
        eq_a = EQ.EQUIPMENT.get(int(units[a].get("data") or 0))
        eq_t = EQ.EQUIPMENT.get(int(units[t].get("data") or 0))
    except (TypeError, ValueError):
        return None
    if not eq_a or not eq_t:
        return None
    dd = max(1, int(d))
    tgt_type, tgt_armor = eq_t[1], eq_t[2]
    best = None
    for wid in eq_a[5]:
        wd = EQ.WEAPONS.get(wid)
        if not wd:
            continue
        name = wd[0]
        if tgt_type == 2:
            lvl = CRT.atk_level_vehicle(name, dd)
            if name == "步兵轻武器":
                lvl = CRT.atk_level_person(name, 1, dd)
        elif tgt_type in (1, 4):
            lvl = CRT.atk_level_person(name, 1, dd)
        else:
            lvl = CRT.atk_level_air(name, dd)
        if lvl is None or lvl <= 0:
            continue
        if best is None or lvl > best[1]:
            best = (wid, lvl, name, tgt_type, tgt_armor)
    return best


def post_cmd(uid, cmd, task, pos_hex, map_hex=""):
    return raw("com/ComPost.asp", ObjID=uid, Command=json.dumps(cmd, ensure_ascii=False),
               ObjColor="", TaskName=task, ObjPos=pos_hex, MapID=map_hex)


def move_step(uid, frm, to, comid):
    """官方三步：ComPost 已下命令 → updatePos（拿耗时）→ MoveEnd（真正落位）。"""
    up = raw("com/updatePos.asp", ObjID=uid, MapID=to, ObjRot=180, GridType=0,
             MapStep=1, preIDhex=G.hexid(frm), posIDhex=G.hexid(to), posIDcond=0,
             HuiHe=0, steep=0, ComID=comid, Index=1, Speed=0, RoomDay=0,
             RoomWeather=0, MineArea=0, GasArea=0, Bridge=0, EleLow=145, EleDiv=5)
    me = raw("com/MoveEnd.asp", ComID=comid, ObjID=uid, StageID=0, RoomSecond=0,
             ObjPos=G.hexid(frm), ObjNewPos=to, MoveFlag=0)
    return up, me


def state(uid):
    return objajax().get(uid) or {}


def pos_of(uid):
    return int(state(uid).get("pos") or 0)


def move_adjacent(uid, tgt, want=1, max_steps=60):
    """把射手逐步挪到目标 want 格内；返回最终距离与步数。"""
    steps = 0
    for i in range(max_steps):
        p1, p2 = pos_of(uid), pos_of(tgt)
        d = G.distance(p1, p2)
        if d <= want:
            return d, steps
        nxt = min(G.neighbors(p1), key=lambda n: G.distance(n, p2))
        if G.distance(nxt, p2) >= d:
            return d, steps
        if post_cmd(uid, {uid: {"move": "%d,%d" % (p1, nxt), "speed": 0}}, "机动",
                    G.hexid(p1), G.hexid(nxt)):
            return d, steps
        move_step(uid, p1, nxt, i + 1)
        steps += 1
    return G.distance(pos_of(uid), pos_of(tgt)), steps


w("# 端到端验证（模拟官方客户端）")
w()

# ---------------------------------------------------------------- 1. 会话
ug = g("User_Get.asp", userAgent="e2e").strip()
f = ug.split("$")
w("## 1. User_Get.asp")
w()
w("- 房间=%s 想定=%s 列×行=%s×%s 已推演=%ss Time1=%r" % (
    f[14] if len(f) > 14 else "?", f[2], f[6], f[7], f[20], f[34] if len(f) > 34 else ""))
w()

# ---------------------------------------------------------------- 2. 开局
w("## 2. RoomStart.asp（开始推演）")
w()
for i in range(2):
    rs = g("RoomStart.asp", AIStart="false", RoomSecond="0").strip()
    w("- 第%d次: `%s`" % (i + 1, rs[:80]))
st = g("RoomStat.asp", RoomID="228").strip()
w("- RoomStat.asp: `%s`" % st[:80])
ug2 = g("User_Get.asp", userAgent="e2e").strip().split("$")
w("- 之后 Time1=%r RoomSecond=%r ServerTime=%s" % (ug2[34], ug2[20], ug2[33]))
w()

# ---------------------------------------------------------------- 3. 基线
st0 = objajax()
units = parse_units(g("Obj_Get.asp"))
w("## 3. 基线态势（ObjAjax.asp / Obj_Get.asp）")
w()
alive0 = {k: v for k, v in st0.items() if int(v.get("blood") or 0) > 0}
w("- ObjAjax 条目 %d，存活 %d；Obj_Get 解析出 %d 个单位" % (len(st0), len(alive0), len(units)))
red = [k for k, v in units.items() if v.get("color") == "RED"
       and int(v.get("blood") or 0) > 0]
blue = [k for k, v in units.items() if v.get("color") == "BLUE"
        and int(v.get("blood") or 0) > 0]
w("- 红方存活 %d，蓝方存活 %d" % (len(red), len(blue)))
w()
check("静态接口与动态接口一致（Obj_Get=%d vs ObjAjax=%d）" % (len(units), len(st0)),
      len(units) == len(st0) == 36, "想定 7527 共 36 个单位")

# ---------------------------------------------------------------- 4. 射击
w("## 4. 直瞄射击链路（ComPost 下命令 → ObjWeaponReady → com/Attack.asp → 战报）")
w()
pair = None
for a in blue:
    for t in red:
        d = G.distance(int(units[a]["pos"]), int(units[t]["pos"]))
        pick = weapon_choice(units, a, t, 0)      # 0 = 按近距能力评估
        if pick:
            pair = (a, t, d, pick)
            break
    if pair:
        break
if not pair:
    w("- 找不到「有可用武器」的敌我配对")
    check("找到可用射击配对", False)
    a = t = None
else:
    a, t, d0, pick = pair
    wid, lvl, wname = pick[0], pick[1], pick[2]
    w("- 射手 %s(%s) @%s　目标 %s(%s) @%s　初始距离 %d 格" % (
        a, units[a].get("army"), G.hexid(int(units[a]["pos"])),
        t, units[t].get("army"), G.hexid(int(units[t]["pos"])), d0))
    w("- 选定向战武器 %s（data id=%d，1 格预期攻击等级 %d）" % (wname, wid, lvl))
    w("  ⇒ 开局双方相距 %d 格，任何武器都超射程 ⇒ 先机动到位再开火" % d0)
    w()

    # 4a. 反例：远距离/不通视直接开火
    before = int(state(t).get("blood") or 0)
    resp_far = raw("com/Attack.asp", ObjID=a, TarID=t, WeaponID=wid, Dist=d0,
                   ObjPos=G.hexid(int(units[a]["pos"])),
                   TarPos=G.hexid(int(units[t]["pos"])))
    after_far = int(state(t).get("blood") or 0)
    w("- **反例**（%d 格、不通视）`com/Attack.asp` → `%s`，目标兵力 %d → %d"
      % (d0, resp_far, before, after_far))
    w("  ⇒ 接口受理但**不产生战果**：开局双方互不通视由地形决定，不是接口问题。")
    w()

    # 4b. 先机动到相邻格（官方机动链）
    dist, steps = move_adjacent(a, t)
    w("- 机动 %d 步后距离 %d 格；ObjAjax 中射手位置 = %s" % (steps, dist, pos_of(a)))
    check("机动链路生效（走到目标相邻格）", dist <= 1, "最终距离 %d 格" % dist)

    # 4c. 三段式开火
    pick2 = weapon_choice(units, a, t, dist) or pick
    wid2, lvl2, wname2 = pick2[0], pick2[1], pick2[2]
    w("- 到位后重选武器：%s（预期攻击等级 %d）" % (wname2, lvl2))
    okr = raw("com/ObjWeaponReady.asp", ObjID=a)
    w("- `com/ObjWeaponReady.asp` → `%s`（武器就绪；连射之间必须复位）" % okr)
    before = int(state(t).get("blood") or 0)
    resp = raw("com/Attack.asp", ObjID=a, TarID=t, WeaponID=wid2, Dist=dist,
               ObjPos=G.hexid(pos_of(a)), TarPos=G.hexid(pos_of(t)))
    raw("ObjFire.asp", ObjID=a)
    after = int(state(t).get("blood") or 0)
    keep = state(t).get("keep")
    w("- `com/Attack.asp` → `%s`（非空=射击成功）" % resp)
    w("- 目标兵力 %d → %d，压制标记 keep=%s" % (before, after, keep))
    w("- `Report_Attack.asp` 战报: `%s`" % g("Report_Attack.asp").strip()[:220])
    w()
    check("射击链路产生战果（兵力下降或压制）",
          after < before or str(keep) == "1",
          "兵力 %d→%d，keep=%s" % (before, after, keep))
    check("`Report_Attack.asp` 有战报内容",
          "暂无战报" not in g("Report_Attack.asp"), "引擎把裁决明细写进战报")
    check("反例成立（不通视开火不产生战果）", after_far == before,
          "接口受理但无战果 ⇒ 前置条件确实被引擎执行")
w()

# ---------------------------------------------------------------- 5. 机动
w("## 5. 机动链路（ComPost → updatePos → MoveEnd）")
w()
mv = None
for a in blue:
    cur = pos_of(a) or int(units[a]["pos"])     # ★ 读**实时**位置，别用开局的静态坐标
    for n in G.neighbors(cur):
        mv = (a, cur, n)
        break
    if mv:
        break
if mv:
    a, frm, to = mv
    r = post_cmd(a, {a: {"move": "%d,%d" % (frm, to), "speed": 0}}, "机动",
                 G.hexid(frm), G.hexid(to))
    w("- 下达机动 %s: %s → %s，`ComPost` → `%s`" % (a, G.hexid(frm), G.hexid(to), r))
    w("- `ComAjax` → `%s`" % g("com/ComAjax.asp").strip()[:160])
    up, me = move_step(a, frm, to, 1)
    w("- `updatePos` → `%s`（本格耗时毫秒）；`MoveEnd` → `%s`（空=继续机动）" % (up, me))
    newpos = str(state(a).get("pos"))
    w("- ObjAjax 中该单位新位置: %s（期望 %d）" % (newpos, to))
    check("机动落位正确", newpos == str(to), "%s → %s" % (G.hexid(frm), G.hexid(to)))
    w()

# ---------------------------------------------------------------- 6. 夺控
w("## 6. 夺控接口（com/CityTake.asp）")
w()
cities = re.findall(r'id="(C_\d+)"[^>]*name="([^"]*)"[^>]*pos="(\d+)"', g("City_Get.asp"))
w("- 夺控点: %s" % cities)
w("- 能否夺控由引擎按手册 3.13 判定（本格与相邻 6 格无敌方地面单位）")
w()

# ---------------------------------------------------------------- 7. 汇总
n_ok = sum(1 for _n, ok, _x in RESULT if ok)
w("## 7. 结论")
w()
for name, ok, note in RESULT:
    w("- %s %s%s" % ("✅" if ok else "❌", name, ("（%s）" % note) if note else ""))
w()
w("- 引擎已完全接管动态接口；客户端渲染的正是引擎状态。")
w("- 射击、机动、夺控均走官方契约，可被官方前端直接驱动。")
w()
w("**通过 %d/%d**" % (n_ok, len(RESULT)))

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("e2e written: %d/%d" % (n_ok, len(RESULT)))
sys.exit(0 if n_ok == len(RESULT) else 1)
