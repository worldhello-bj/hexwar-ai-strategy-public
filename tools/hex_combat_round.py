# -*- coding: utf-8 -*-
"""战斗数据采集：开一局 -> 把交战双方开进射程 -> 按射程选武器开火 -> 追踪战果 -> 提前收工。

与 hex_play_round.py 的区别：这里**以产生战果为目标**，不是被动观察。
上一版失败的原因已记录：`weapon` 必须按射程选（index 0 常是 1~10 格的近程武器，
19 格外打不出战果），所以这里会先用 `engine/equip.py` 的射程表挑武器，打不出战果就换下一个。

产物：7527_transport_capture/combat_round_<时间戳>/
    timeline.jsonl  每次采样一条（单位位置/兵力/压制/弹药、夺控点、三段计分、战报、RoomSecond）
    actions.jsonl   每个动作的请求参数与响应原文
    combat.json     交战汇总（每发的结果、武器索引、时耗、计分变化）
    report.md       可读报告
"""
import argparse
import io
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hex_transport_capture as H                       # noqa: E402
from engine import geometry as GEO, equip as EQ         # noqa: E402
from engine import paths as _paths                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from engine import crt as CRT                          # noqa: E402
from engine import Game as _Game                        # noqa: E402

GAME = _Game(seed=20261015)     # 只为拿高程表与通视判定（加载 snapshot/parsed/map_hexes.json）


def elev(mapid):
    try:
        return GAME.elevation(int(mapid))
    except Exception:                                               # noqa: BLE001
        return 0


def dh_m(a, b):
    """两格高差（米）。手册 4.4 / 表3 用的就是它。"""
    return elev(a) - elev(b)


def los_ok(a, b):
    """通视判定（手册 4.4：高看低、观察点高程 +1 等级、连线插值）。不通视就打不出战果。"""
    try:
        ok, _ = GAME.los(int(a), int(b))
        return ok
    except Exception:                                               # noqa: BLE001
        return True

# 采集数据已封存到 archive/exploration/（见该目录 README）；
# 未封存的检出或 train/ 自包含副本里，老路径仍然存在 —— 两者都支持。
CAP = os.path.join(ROOT, "archive", "exploration", "capture", "7527_transport_capture")
if not os.path.isdir(CAP):
    CAP = os.path.join(ROOT, "7527_transport_capture")


def new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": H.UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    return s


def login_as(user, pwd):
    H.USERNAME, H.PASSWORD = user, pwd
    H.S = new_session()
    return H.S, H.login()


def jget(sess, base7, path, **params):
    r = sess.get(base7 + path, params=params or None, timeout=30,
                 headers={"Referer": base7 + "main.html"})
    return H.decode(r.content)


def in_room(sess, base7):
    parts = jget(sess, base7, "User_Get.asp", userAgent="cb").split("$")
    return (len(parts) >= 21 and parts[5].strip() not in ("", "-1"), parts)


def quit_room(sess, uid):
    from urllib.parse import urlsplit
    c = requests.cookies.create_cookie(name="UserID", value=uid or "",
                                       domain=urlsplit(H.HOST).hostname, path="/")
    sess.cookies.set_cookie(c)
    try:
        return sess.get(H.BASE + "SetUp/SetUp_Transfer/Quit.asp", timeout=30)
    finally:
        sess.cookies.clear(urlsplit(H.HOST).hostname, "/", "UserID")


def snapshot(sess, base7):
    parts = jget(sess, base7, "User_Get.asp", userAgent="cb").split("$")
    units = H.parse_obj_get(jget(sess, base7, "Obj_Get.asp"))
    dyn = H.parse_obj_ajax(jget(sess, base7, "ObjAjax.asp"))
    rep = " ".join(jget(sess, base7, "Report_All.asp").split())
    atk = " ".join(jget(sess, base7, "Report_Attack.asp").split())
    city = jget(sess, base7, "City_Get.asp")
    return {"ts": time.strftime("%H:%M:%S"),
            "room_second": parts[20] if len(parts) > 20 else "?",
            "start_time": parts[3] if len(parts) > 3 else "",
            "units": units, "dynamic": dyn, "report_all": rep[:1500],
            "report_attack": atk[:1500],
            "cities": re.findall(r'id="(C_[^"]*)"[^>]*name="([^"]*)"[^>]*pos="(\d+)"[^>]*flag="([^"]*)"', city)}


def cmd(sess, base7, objid, color, command, task, pos_hex, map_hex=""):
    params = {"ObjID": objid, "Command": json.dumps(command, ensure_ascii=False),
              "ObjColor": color, "TaskName": task, "ObjPos": pos_hex, "MapID": map_hex}
    body = jget(sess, base7, "com/ComPost.asp", **params)
    echo = jget(sess, base7, "com/ComAjax.asp")
    m = re.search(r"(\d+)\|[^|]*\|", echo)
    return {"params": params, "com_post": body, "com_ajax": echo[:300],
            "com_id": m.group(1) if m else ""}


MAP_FILE = os.path.join(_paths.snapshot_dir(), "map_hexes.json")
try:
    MAPINFO = json.load(io.open(MAP_FILE, encoding="utf-8"))
except Exception:                                                   # noqa: BLE001
    MAPINFO = {}


def cell(mid):
    """取某格的静态属性（贴图 hex / cond / hextype / step）。"""
    return MAPINFO.get(str(int(mid))) or MAPINFO.get(str(mid)) or {}


def move_step(sess, base7, unit, frm, to, color):
    """向指定格走一步（ComPost -> updatePos -> MoveEnd）。

    ★ 参数真值（2026-10-15 从官方客户端实际发出的请求抓取，已核实）：

        com/updatePos.asp?ObjID=<单位>&MapID=<目标格 mapid>&ObjRot=180
          &GridType=<目标格 hextype>&MapStep=<目标格 step>
          &preIDhex=<起格 hex 属性>&posIDhex=<目标格 hex 属性>&posIDcond=<目标格 cond>
          &HuiHe=1&steep=0&ComID=0&Index=1&Speed=0&RoomDay=0&RoomWeather=0
          &MineArea=0&GasArea=0&Bridge=0&EleLow=145&EleDiv=5

    即：`MapID` 是 **mapid**（地图格 div 的 id 就是 mapid）；`preIDhex`/`posIDhex`
    是格的 **`hex` 属性（地形贴图编号）**，不是格号。把格号填进去会让服务端把单位
    存成错误的 pos（踩过）。
    """
    c = cmd(sess, base7, unit, color,
            {unit: {"move": "%d,%d" % (frm, to), "speed": 0}}, "机动",
            GEO.hexid(frm), GEO.hexid(to))
    tc = cell(to)
    ms = jget(sess, base7, "com/updatePos.asp", ObjID=unit, MapID=to, ObjRot=180,
              GridType=tc.get("hextype", 1), MapStep=tc.get("step", 1),
              preIDhex=cell(frm).get("hex", 0), posIDhex=tc.get("hex", 0),
              posIDcond=tc.get("cond", 0), HuiHe=1, steep=0, ComID=0,
              Index=1, Speed=0, RoomDay=0, RoomWeather=0, MineArea=0, GasArea=0, Bridge=0,
              EleLow=145, EleDiv=5)
    # ★ 第三步 MoveEnd 才是"提交这一步"；其 ObjPos/ObjNewPos 必须是 **mapid**。
    #   受控实验（tools/hex_move_probe.py）：传格号 → 服务端把 pos 存成格号（如 3435）；
    #   传 mapid → pos 正确（如 120035）。这是历次"机动位置存错"的根因。
    me = jget(sess, base7, "com/MoveEnd.asp", ComID=c["com_id"] or 1, ObjID=unit,
              StageID=0, RoomSecond=0, ObjPos=frm, ObjNewPos=to, MoveFlag=0)
    return {"kind": "move", "unit": unit, "frm": frm, "to": to,
            "update_pos": ms.strip(), "move_end": me.strip(), "com_post": c["com_post"],
            "cell": tc}


def pos_of(sess, base7, unit):
    dyn = H.parse_obj_ajax(jget(sess, base7, "ObjAjax.asp"))
    d = dyn.get(unit) or {}
    try:
        return int(d.get("pos"))
    except (TypeError, ValueError):
        return None


def positions(sess, base7, ids):
    """一次 ObjAjax 取多个单位的位置（比分次 pos_of 快）。"""
    dyn = H.parse_obj_ajax(jget(sess, base7, "ObjAjax.asp"))
    out = {}
    for i in ids:
        try:
            out[i] = int((dyn.get(i) or {}).get("pos"))
        except (TypeError, ValueError):
            out[i] = None
    return out


def _dec_cell(b):
    """Caijue 页是 GBK 标签 + UTF-8 值混编，按单元格各自试编码。"""
    b = b.strip()
    for enc in ("utf-8", "gb18030"):
        try:
            return b.decode(enc)
        except Exception:                                 # noqa: BLE001
            pass
    return b.decode("gb18030", "replace")


def parse_caijue(raw):
    """把 Caijue.asp 字节解析成字段 dict（按 td 位置）。"""
    cells = [_dec_cell(b) for b in re.findall(rb"<td[^>]*>(.*?)</td>", raw, re.S)]

    def g(i):
        return cells[i] if i < len(cells) else ""

    return {
        "cell_count": len(cells),
        "射击单位": g(1), "目标单位": g(3), "武器": g(5), "距离": g(7),
        "高度差": g(9), "攻击等级": g(11), "随机数": g(13), "战斗结果": g(15),
        "射击方状态": g(17), "目标状态": g(19), "修正前": g(21), "装甲级别": g(23),
        "修正结果": g(25), "最终战果": g(27),
        "cells": cells,
    }


def weapons_of(units, unit):
    """从静态 Wz 字段拿该单位的武器 data 列表。"""
    wz = (units.get(unit) or {}).get("wz") or ""
    ids = [int(x) for x in re.findall(r"\d+", wz)]
    return ids


def expected_level(units, shooter, target, weapon_name, dist, dh=0):
    """用裁决表（表1/表2/表9 + 表3 高差修正）估这门炮能拿到多少攻击等级；越高越好。"""
    t = units.get(target) or {}
    tid = str(t.get("typeid") or "")
    try:
        n = max(1, min(4, int((units.get(shooter) or {}).get("blood") or 1)))
    except (TypeError, ValueError):
        n = 1
    try:
        if tid == "2":
            lvl = CRT.atk_level_vehicle(weapon_name, dist)
        elif tid == "3":
            lvl = CRT.atk_level_air(CRT.CRT_AIR_NAME.get(weapon_name, weapon_name), dist)
        else:
            lvl = CRT.atk_level_person(weapon_name, n, dist)
    except Exception:                                               # noqa: BLE001
        return None
    if lvl is None:
        return None
    return lvl + CRT.atk_level_elev_fix(abs(int(dh)), dist)


def best_weapon(units, shooter, target, dist, skip=(), dh=0):
    """返回 (索引, 武器ID, (名, 弹药, 射程), 攻击等级)；优先攻击等级高的，其次射程长的。

    超出射程或对不上目标类型的武器攻击等级为 None，会被排在后面。
    """
    ids = weapons_of(units, shooter)
    rows = []
    for i, wid in enumerate(ids):
        if i in skip:
            continue
        info = EQ.weapon(wid) or ("?", "?", 0)
        lvl = expected_level(units, shooter, target, info[0], dist, dh)
        rows.append((lvl is not None, lvl if lvl is not None else -99, info[2], i, wid, info, lvl))
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
    r = rows[0]
    return r[3], r[4], r[5], r[6]


def fire(sess, base7, unit, target, weapon_id, color, rec, track=0):
    """按客户端真实协议开火：**直调 com/Attack.asp**（不是 ComPost）。

    证据（war.min.js `CommandAttack`）：
      com/Attack.asp?ObjID&TarID&ObjPos&TarPos&ObjType&TarType&Dist
        &ObjEle&TarEle&TarHex&TarHexType&WeaponID&TarStack&StageID&GuideID
        &ComID&RoomSecond&ObjA0&ObjF3&AttackRot&RoomDay&RoomWeather&Time1
    返回非空 = 成功；战果经下一帧 ObjAjax 下发。`WeaponID` 是武器 **data ID**。
    """
    u = rec["units"].get(unit) or {}
    t = rec["units"].get(target) or {}
    m = int(u.get("pos") or 0)
    k = int(t.get("pos") or 0)
    dyn = (rec.get("dynamic") or {}).get(unit) or {}
    stack = 1 if sum(1 for a in rec["units"].values()
                     if str(a.get("pos")) == str(k) and a.get("color") == t.get("color")
                     and str(a.get("blood")) not in ("0", "")) > 1 else 0
    params = {
        "ObjID": unit, "TarID": target, "ObjPos": m, "TarPos": k,
        "ObjType": u.get("typeid"), "TarType": t.get("typeid"),
        "Dist": GEO.distance(m, k),
        "ObjEle": elev(m), "TarEle": elev(k),
        "TarHex": cell(k).get("hex", 0), "TarHexType": cell(k).get("hextype", 1),
        "WeaponID": weapon_id, "TarStack": stack,
        "StageID": 1, "GuideID": "", "ComID": 0, "RoomSecond": 0,
        "ObjA0": dyn.get("a0", u.get("a0", 0)), "ObjF3": u.get("fort", 0),
        "AttackRot": u.get("rot", 0),
        "RoomDay": 0, "RoomWeather": 0, "Time1": "",
    }
    resp = jget(sess, base7, "com/Attack.asp", **params)
    return {"kind": "fire", "unit": unit, "target": target, "weapon_id": weapon_id,
            "track": track, "params": params,
            "attack_resp": resp.strip() if isinstance(resp, str) else resp}


def blood_keep(rec, unit):
    d = rec["dynamic"].get(unit) or {}
    a = rec["units"].get(unit) or {}
    return str(d.get("blood", a.get("blood"))), str(d.get("keep", a.get("keep", "0"))), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, default=7)
    ap.add_argument("--color2", default="RED")
    ap.add_argument("--max-shots", type=int, default=8)
    ap.add_argument("--want-results", type=int, default=3, help="拿到这么多有战果的射击就收工")
    ap.add_argument("--max-steps", type=int, default=40, help="最多机动多少步（用来接近）")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--pos-probe", action="store_true",
                    help="只做受控实验：定死 com/updatePos.asp 里哪个参数是新位置")
    args = ap.parse_args()

    H.guard()
    uA, pA = os.environ.get("HEXWAR_USER", ""), os.environ.get("HEXWAR_PASS", "")
    uB, pB = os.environ.get("HEXWAR_USER2", ""), os.environ.get("HEXWAR_PASS2", "")
    base7 = "%s/WARGAMEAI%d/" % (H.HOST, args.room)
    H.BASE7 = base7
    d = os.path.join(CAP, "combat_round_%s" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    tl = io.open(os.path.join(d, "timeline.jsonl"), "w", encoding="utf-8")
    ac = io.open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8")
    log = []

    def w(x=""):
        log.append(str(x))
        print(x)

    def rec_action(o):
        ac.write(json.dumps(o, ensure_ascii=False) + "\n")
        ac.flush()

    caijue_seen = set()

    def harvest_caijue(text):
        """从战报里找 Caijue(<id>)，抓取并解析该次战斗的裁决明细。"""
        for m in re.finditer(r"Caijue\((\d+)\)", text or ""):
            cid = m.group(1)
            if cid in caijue_seen:
                continue
            caijue_seen.add(cid)
            try:
                rr = sessA.get(base7 + "Caijue.asp", params={"ID": cid}, timeout=30,
                               headers={"Referer": base7 + "main.html"})
            except Exception:                                 # noqa: BLE001
                continue
            if not rr.content:
                continue
            od = os.path.join(d, "caijue")
            os.makedirs(od, exist_ok=True)
            open(os.path.join(od, "caijue_%s.html" % cid), "wb").write(rr.content)
            info = parse_caijue(rr.content)
            info["id"] = cid
            rec_action({"kind": "caijue", "id": cid, "fields": info})
            w("    ◆ 裁决#%s: 攻击等级=%s 随机数=%s 战斗结果=%s 装甲=%s 修正=%s 最终战果=%s"
              % (cid, info["攻击等级"], info["随机数"], info["战斗结果"],
                 info["装甲级别"], info["修正结果"], info["最终战果"]))

    w("# 战斗数据采集 %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    deadline = time.time() + args.minutes * 60

    # ---- 准备一局 ----
    sessA, _ = login_as(uA, pA)
    url = "%s/WARGAMEAI%d/Reset/Post.asp" % (H.HOST, args.room)
    m = re.search(r'<form[^>]*action="([^"]*)"[^>]*>(.*?)</form>',
                  jget(sessA, "%s/WARGAMEAI%d/" % (H.HOST, args.room),
                       "Reset/Post.asp", scenarioId=7527, roomId=args.room), re.S)
    if m:
        data = {}
        for im in re.finditer(r'<textarea[^>]*name="([^"]+)"[^>]*>(.*?)</textarea>',
                              m.group(2), re.S):
            data[im.group(1)] = im.group(2).strip()
        r2 = sessA.post(requests.compat.urljoin(url, m.group(1)), data=data, timeout=30)
        w("- 复位房间 -> %s" % r2.status_code)

    sessA, uidA = login_as(uA, pA)
    quit_room(sessA, uidA)
    time.sleep(0.5)
    H.S = sessA
    H.enter(args.room, "NONE", uidA)
    okA, pA_ = in_room(sessA, base7)
    w("- A(导演 %s) 在座=%s" % (uidA, okA))

    sessB, uidB = login_as(uB, pB)
    quit_room(sessB, uidB)
    time.sleep(0.5)
    H.S = sessB
    H.enter(args.room, args.color2, uidB)
    okB, _ = in_room(sessB, base7)
    w("- B(%s %s) 在座=%s" % (args.color2, uidB, okB))

    started = ""
    for attempt in range(4):
        for v in ({"AIStart": "false", "RoomSecond": "0"},):
            txt = jget(sessA, base7, "RoomStart.asp", **v)
            w("- RoomStart 第%d次 -> %r" % (attempt + 1, txt[:110]))
            if txt.strip() not in ("-1", "0", ""):
                started = txt
                break
            time.sleep(1.5)
        if started:
            break
    if not started:
        w("!!! 开局失败，无法采集战斗数据")
        io.open(os.path.join(d, "report.md"), "w", encoding="utf-8").write("\n".join(log))
        return
    w("- 开局成功 Time1=%s" % started.split(",")[0])

    # ---- 选交战对象：初始距离最近的一对（蓝->红）----
    rec = snapshot(sessA, base7)
    tl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    units = rec["units"]

    if args.pos_probe:
        probe = [k for k, a in sorted(units.items(), key=lambda kv: int(kv[0].split("_")[1]))
                 if a.get("typeid") == "2" and str(a.get("pos") or "0") != "0"]
        if not probe:
            w("没有可用的地面车辆单位")
            return
        unit = probe[0]
        cur = int(units[unit]["pos"])
        w("受控实验：单位 %s (%s) 起始 pos=%d (格号 %s)"
          % (unit, units[unit].get("ico"), cur, GEO.hexid(cur)))
        variants = [
            ("V1 MapID=格号 preIDhex=mapid posIDhex=mapid", "hex", "mapid", "mapid"),
            ("V2 MapID=mapid preIDhex=0 posIDhex=0", "mapid", "zero", "zero"),
            ("V3 MapID=格号 preIDhex=0 posIDhex=0", "hex", "zero", "zero"),
            ("V4 MapID=格号 preIDhex=格号 posIDhex=格号", "hex", "hex", "hex"),
            ("V5 MapID=格号 preIDhex=贴图 posIDhex=贴图", "hex", "tex", "tex"),
        ]
        for name, mid, pre, pos in variants:
            tgt = sorted(GEO.neighbors(cur))[0]
            tc = cell(tgt)

            def val(kind, mid_):
                return {"hex": GEO.hexid(mid_), "mapid": mid_, "zero": 0,
                        "tex": cell(mid_).get("hex", 0)}[kind]

            ms = jget(sessA, base7, "com/updatePos.asp", ObjID=unit, MapID=val(mid, tgt),
                      ObjRot=180, GridType=tc.get("hextype", 1), MapStep=tc.get("step", 1),
                      preIDhex=val(pre, cur), posIDhex=val(pos, tgt),
                      posIDcond=tc.get("cond", 0), HuiHe=0, steep=0, ComID=1, Index=1,
                      Speed=0, RoomDay=0, RoomWeather=0, MineArea=0, GasArea=0, Bridge=0,
                      EleLow=145, EleDiv=5)
            time.sleep(1)
            dyn = H.parse_obj_ajax(jget(sessA, base7, "ObjAjax.asp"))
            got = (dyn.get(unit) or {}).get("pos")
            w("  %-46s 返回=%-6r 服务端 pos=%-8s 期望=%d(%s)  %s"
              % (name, ms.strip()[:6], got, tgt, GEO.hexid(tgt),
                 "✔" if str(got) == str(tgt) else "✘"))
            if str(got).isdigit():
                cur = int(got)
        io.open(os.path.join(d, "report.md"), "w", encoding="utf-8").write("\n".join(log))
        tl.close()
        ac.close()
        print("probe done ->", d)
        return
    blue = [k for k, a in units.items() if a.get("color") == "BLUE" and str(a.get("blood")) != "0"
            and str(a.get("typeid")) in ("1", "2")]
    red = [k for k, a in units.items() if a.get("color") == "RED" and str(a.get("blood")) != "0"
           and str(a.get("typeid")) in ("1", "2")]
    pairs = sorted(((GEO.distance(int(units[a]["pos"]), int(units[t]["pos"])), a, t)
                    for a in blue for t in red), key=lambda x: x[0])

    def reach(u):
        ws = weapons_of(units, u)
        return max([(EQ.weapon(x) or ("?", "?", 0))[2] for x in ws] or [0])

    # ★ 通视是直瞄的硬前提（不同通视肯定打不出战果）。先筛出通视的对，再在其中挑。
    los_pairs = [(dd, a, t) for dd, a, t in pairs
                 if los_ok(units[a]["pos"], units[t]["pos"])]
    w("- 共 %d 对敌我单位，其中**通视**的 %d 对" % (len(pairs), len(los_pairs)))
    pool = los_pairs or pairs
    # ★ 开局多半无通视，必须靠接近；选一对"接近到 6 格后攻击等级最高"的
    scored = []
    for dd, a, t in pool:
        dh = dh_m(units[a]["pos"], units[t]["pos"])
        bw6 = best_weapon(units, a, t, min(dd, 6), dh=dh)
        lvl6 = bw6[3] if bw6 and bw6[3] is not None else -99
        scored.append((-lvl6, dd, a, t, bw6, dh))
    scored.sort()
    _, d0, atk, tgt, bw, dh = scored[0]
    if bw is not None and bw[3] is not None:
        w("- 选到一对：用 %s(预估等级 %s) 打 %s（将接近到 6 格内并建立通视）"
          % (bw[2][0], bw[3], (units[tgt] or {}).get("ico")))
    else:
        w("- 开局无可用攻击等级，选最近的一对并尝试接近")
    w("- 高程：攻击方 %d m，目标 %d m，**高差 %d m**（%s）"
      % (elev(units[atk]["pos"]), elev(units[tgt]["pos"]), dh,
         "目标在低处" if dh > 0 else ("目标在高处" if dh < 0 else "同高")))
    w("- 交战对：%s %s @%s  ↔  %s %s @%s  距离 %d 格"
      % (atk, (units[atk] or {}).get("ico"), GEO.hexid(int(units[atk]["pos"])),
         tgt, (units[tgt] or {}).get("ico"), GEO.hexid(int(units[tgt]["pos"])), d0))
    w("- 双方武器：%s=%s   %s=%s"
      % (atk, [(i, EQ.weapon(x)[0], EQ.weapon(x)[2]) for i, x in enumerate(weapons_of(units, atk))],
         tgt, [(i, EQ.weapon(x)[0], EQ.weapon(x)[2]) for i, x in enumerate(weapons_of(units, tgt))]))

    want = 1          # ★ 贴近到相邻格：手册 4.4「相邻格必通视」→ 直瞄硬前提必然满足
    steps = 0
    # ---- 接近：双方交替各走一步，直到相邻 - - --
    while (time.time() < deadline and steps < args.max_steps
           and (d0 > want
                or not los_ok(units[atk]["pos"], units[tgt]["pos"]))):
        ps = positions(sessA, base7, (atk, tgt))
        for who, foe in ((atk, tgt), (tgt, atk)):
            frm, fpos = ps.get(who), ps.get(foe)
            if frm is None or fpos is None:
                continue
            if GEO.distance(frm, fpos) <= want:
                continue
            nbrs = sorted(GEO.neighbors(frm), key=lambda n: GEO.distance(n, fpos))
            col = (units[who] or {}).get("color") or ""
            st = move_step(sessA, base7, who, frm, nbrs[0], col)
            st["kind"] = "approach"
            st["dist_before"] = GEO.distance(frm, fpos)
            rec_action(st)
            steps += 1
            w("  ~ 接近 %s %s→%s (updatePos=%s, 距离 %d→%d)"
              % (who, GEO.hexid(frm), GEO.hexid(nbrs[0]), st["update_pos"], st["dist_before"],
                 GEO.distance(nbrs[0], fpos)))
        ps = positions(sessA, base7, (atk, tgt))
        if ps.get(atk) and ps.get(tgt):
            d0 = GEO.distance(ps[atk], ps[tgt])
            units.setdefault(atk, {})["pos"] = str(ps[atk])
            units.setdefault(tgt, {})["pos"] = str(ps[tgt])
        if steps % 4 == 0:
            rec = snapshot(sessA, base7)
            rec["note"] = "approach %d" % steps
            tl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tl.flush()
    w("- 机动 %d 步后距离 %d 格（目标 %d 格）" % (steps, d0, want))

    # ---- 开火并追踪战果 ----
    combat = {"attacker": atk, "target": tgt, "steps": steps, "shots": [], "results": 0,
              "kills": 0, "killed": []}
    shots = 0
    tried = set()
    time.sleep(2)
    while time.time() < deadline and shots < args.max_shots and combat["results"] < args.want_results:
        rec = snapshot(sessA, base7)
        units = rec["units"]
        if atk not in units or tgt not in units:
            w("- 交战方不在表内，收工")
            break
        dist = GEO.distance(int(units[atk]["pos"]), int(units[tgt]["pos"]))
        dh = dh_m(units[atk]["pos"], units[tgt]["pos"])
        los = los_ok(units[atk]["pos"], units[tgt]["pos"])
        b0, k0, d0_ = blood_keep(rec, tgt)
        ammo0 = {k: d0_.get(k) for k in ("a3", "a4", "a5")}
        idx, wid, info, lvl = (bw if (bw := best_weapon(units, atk, tgt, dist, skip=tried,
                                                       dh=dh)) else
                               (None, None, None, None))
        if idx is None:
            w("- %s 已无未试武器，收工" % atk)
            break
        w("- 第 %d 发：%s -> %s  距离 %d 格  高差 %d m  通视=%s  用武器[%d]=%s(弹药%s,射程%d格,预估等级%s)  目标兵力=%s 压制=%s"
          % (shots + 1, atk, tgt, dist, dh, los, idx, info[0], info[1], info[2], lvl, b0, k0))
        f = fire(sessA, base7, atk, tgt, wid, (units[atk] or {}).get("color") or "", rec)
        rec_action(f)
        shots += 1
        # 追踪：最多 20 秒内的变化（攻击间隔约 10 秒）
        change = None
        for _ in range(10):
            time.sleep(2)
            r2 = snapshot(sessA, base7)
            tl.write(json.dumps(r2, ensure_ascii=False) + "\n")
            tl.flush()
            harvest_caijue(r2["report_attack"])
            b1, k1, d1 = blood_keep(r2, tgt)
            if b1 != b0 or k1 != k0:
                change = {"blood": [b0, b1], "keep": [k0, k1],
                          "ammo": [ammo0, {k: d1.get(k) for k in ("a3", "a4", "a5")}],
                          "report_attack": r2["report_attack"][:300],
                          "report_all": r2["report_all"][:300],
                          "room_second": r2["room_second"]}
                break
        if change:
            combat["results"] += 1
            if str(change["blood"][1]) == "0":
                combat["kills"] += 1
                combat["killed"].append(tgt)
                w("  ★ 战果：兵力 %s→%s，压制 %s→%s（**歼灭**）"
                  % (change["blood"][0], change["blood"][1], change["keep"][0], change["keep"][1]))
            else:
                w("  ★ 战果：兵力 %s→%s，压制 %s→%s"
                  % (change["blood"][0], change["blood"][1], change["keep"][0], change["keep"][1]))
            if change["report_attack"]:
                w("    战报：%s" % change["report_attack"][:160])
        else:
            w("  - 20 秒内无可见战果（武器可能仍不够或未命中）")
        combat["shots"].append({"idx": idx, "weapon": info[0], "ammo": info[1],
                                "range": info[2], "level": lvl, "dist": dist,
                                "change": change})
        if not change:
            tried.add(idx)                       # 这门炮无效，下一发换一门
            continue
        # 打完一发后换到对方视角打一发，拿到双方数据
        if combat["kills"] == 0 and str(change["blood"][1]) != "0":
            rec = snapshot(sessA, base7)
            units = rec["units"]
            dist2 = GEO.distance(int(units[tgt]["pos"]), int(units[atk]["pos"]))
            idx2, wid2, info2, lvl2 = (bw2 if (bw2 := best_weapon(
                units, tgt, atk, dist2,
                dh=dh_m(units[tgt]["pos"], units[atk]["pos"]))) else
                (None, None, None, None))
            if idx2 is not None:
                b0b, k0b, _ = blood_keep(rec, atk)
                w("- 反击：%s -> %s  距离 %d 格  用武器[%d]=%s(射程%d格,预估等级%s)  目标兵力=%s"
                  % (tgt, atk, dist2, idx2, info2[0], info2[2], lvl2, b0b))
                f2 = fire(sessA, base7, tgt, atk, wid2, (units[tgt] or {}).get("color") or "", rec)
                rec_action(f2)
                for _ in range(8):
                    time.sleep(2)
                    r3 = snapshot(sessA, base7)
                    tl.write(json.dumps(r3, ensure_ascii=False) + "\n")
                    tl.flush()
                    b1b, k1b, _ = blood_keep(r3, atk)
                    if b1b != b0b or k1b != k0b:
                        combat["results"] += 1
                        w("  ★ 反击战果：兵力 %s→%s，压制 %s→%s" % (b0b, b1b, k0b, k1b))
                        break

    combat["shots_fired"] = shots
    io.open(os.path.join(d, "combat.json"), "w", encoding="utf-8").write(
        json.dumps(combat, ensure_ascii=False, indent=1))
    tl.close()
    ac.close()

    w("")
    w("## 汇总")
    w("- 机动 %d 步；射击 %d 发；**有战果 %d 次**；歼灭 %d 个" % (steps, shots, combat["results"],
                                                             combat["kills"]))
    for i, s in enumerate(combat["shots"], 1):
        c = s["change"]
        w("  %d) 距离%d格 武器[%d]=%s(射程%d) -> %s"
          % (i, s["dist"], s["idx"], s["weapon"], s["range"],
             ("兵力 %s→%s 压制 %s→%s" % (c["blood"][0], c["blood"][1], c["keep"][0], c["keep"][1]))
             if c else "无战果"))
    w("- 产物：%s" % d)
    io.open(os.path.join(d, "report.md"), "w", encoding="utf-8").write("\n".join(log))
    print("ok ->", d)


if __name__ == "__main__":
    main()
