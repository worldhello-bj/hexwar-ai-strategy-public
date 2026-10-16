# -*- coding: utf-8 -*-
"""到**实际服务器**打一发巡飞弹，并**全程监听**服务端回话（手册 4.10 的未确证项）。

要回答的问题（`docs/巡飞弹生命周期.md` §4 / `docs/未确证裁决表.md`）：

1. **部署**：平台侧“上车”是不是 `com/ObjInto.asp`？上车后弹的字段长什么样
   （`pos/pos2/blood/color/car/sons`）？**过了第 0 秒还能不能上车**（手册 4.10(3)）？
2. **发射**：`{载车: {"off": 3}}` + `com/Obj2_Out.asp` 是不是发射？第 2 段要哪些参数（`Time1`）？
3. **发射后服务端下发什么**：弹的 `space`、`radar`、`car`、`stop`、`pos`；载车的 `sons/stop`。
4. **巡飞计时**：客户端公式是 `FlyTime = parseInt(1200 / RoomSpeed)`，超时后由**客户端**
   发 `CommandKill`；服务端自己会不会到点销毁？（本探针是个 Python 客户端，
   不带那套 JS 计时器 —— 所以“多久后消失”直接回答了“服务器是否自己计时”。）
5. 载车被摧毁时在飞的弹是否消失（手册 4.10(5)）—— 由 `--kill-carrier` 开关控制，默认**不做**。

## 实测结论（2026-10-16，练习服 `www.hexwar.cn` 推演室2）

★ 先记住平台的房间机制（踩过的坑）：

- 房间刚 `初始化` 或刚 `SetUpSave` 进房后，`RoomStart.asp` 会**短暂**回 `-1`（几分钟后即成功）
  ⇒ 工具里带 6 次重试；
- **只有 导演 一人、无人方也无 AI 时，服务端会下发 `{COM:{"pause":1}}`（“推演暂停”），
  时钟 `RoomSecond` 恒 0**，此时指令被挂在队列里（`ObjInto` 回 `False`）；
  **让一个玩家入座**（`SetUpSave.asp` + `GameColor=RED`，`UserStart` 变 1）后时钟就开始走
  （实测约 1 秒 sim / 1 秒真实）—— 这就是 `--seat` 的用途。

实测到的协议与字段：

| 步骤 | 请求 | 响应 | 状态变化 |
|---|---|---|---|
| 部署 | 第1段 `com/ComPost.asp?ObjID=<弹>&Command={"<弹>":{"inside":"<车>"}}&TaskName=上车` → 第2段 `com/ObjInto.asp?ObjID=<弹>&CarID=<车>&ComID=<id>` | 第1段 `''`（受理）；第2段 **`True`** | 弹 `pos=0 pos2=<载具格> blood=0 color='' car=Obj_0`；载车 `sons=1` |
| 发射 | 第1段 `com/ComPost.asp?ObjID=<车>&Command={"<车>":{"off":3}}&TaskName=发射` → 第2段 `com/Obj2_Out.asp?ObjID=<车>&ObjTypeID=3&Time1=&ComID=<id>&ObjStack=0` | 第1段 `''`；第2段 **`True`** | 载车 **`sons=0` + `stop=1`**；弹仍是 `pos=0 pos2=<载具格> blood=0 color='' car=Obj_0`（**不恢复**） |

★ 三条与手册/本仓库模型有关的结论：

1. **平台不强制“只能对抗开始前部署”**：实测在 `RoomSecond=75` 上车照样 `True`
   ⇒ 手册 4.10(3) 是我们默认 `deploy_window="opening"` 的**规则来源**，
   而平台的宽松行为对应 `deploy_window="any"`（两者都保留，默认忠于手册）。
2. **发射后服务端不下发 `radar`**（实测 240 秒内 `radar` 一直为空），
   弹也**不变成图上的独立单位**（`pos` 仍 0、`blood` 仍 0、`color` 仍空）；
   “在飞”在服务端只体现为**载车 `stop=1`**、`sons` 归 0。
   ⇒ 巡飞计时是**客户端自己记账**（`FlyTime=parseInt(1200/RoomSpeed)`）后发 `CommandKill`；
   本引擎把“上限 400 秒自毁”实现成服务端权威计时，是对客户端行为的**建模**（已在文档标注）。
3. `space=-1` 是**巡飞弹的静态类型标记**（同格无人机 `YY4` 是 `space=0`、载车是 `space=1`），
   不是“在飞”标志 —— 客户端拿它筛出巡飞弹、再用自己的计时器判断该不该销毁。
   另外弹的法定载车由载车自己的 **`son` 编制表**给出（ZBD100 = `PLA2-191,VUT5,UGVS,RDOG,YY4,WS43`）；
   装到同格没有该编制的车上（如 UGVS）第 2 段直接回 `False`。

## 安全约定

- 凭据只从环境变量读（`HEXWAR_USER`/`HEXWAR_PASS` 或 `HEXWAR_USER2`/`HEXWAR_PASS2`），
  绝不写盘、绝不打印；主机用 `--host` 或 `HEXWAR_BASE`（默认练习服）。
- **默认只读**：不带开关时只登录、读大厅里该房间的状态、给出“打算怎么做”。
- 会改状态的开关：`--enter`（进房）、`--start`（开局）、`--load`（部署上车）、
  `--launch`（发射）、`--kill-carrier`（额外打掉发射车，默认关）。
- 请只在自己**空闲**的推演室上跑（`--room` 先看大厅里的状态码 0；本探针每次都会先读一次大厅）。
- 原始请求/响应与时间线写在 `analysis/missile_probe/<时间戳>/`（**gitignore**，不入库）。

用法::

    py -3.13 tools\\hex_missile_probe.py --as b --room 2                   # 只读：看打算做什么
    py -3.13 tools\\hex_missile_probe.py --as b --room 2 --enter --start \\
        --load --launch --poll-seconds 240
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hex_opening_check as OC                       # noqa: E402

OUTROOT = os.path.join(ROOT, "analysis", "missile_probe")
MISSILE_ICOS = ("WS43",)                             # 想定 7527 的巡飞弹图标
MISSILE_DATA = ("389",)                              # engine/equip.py：装备 389 = 巡飞弹
USER_GET_FIELDS = ["CenterID", "GridDiff", "Moban", "StartTime", "UserID", "RoomID",
                   "HexCols", "HexRows", "OnTime", "ProMode", "GameColor", "WarDateTime",
                   "GridMeter", "RecordID", "RoomTitle", "UserStart", "RoomSpeed",
                   "FirstColor", "Ditu", "AllTime", "RoomSecond", "TeamName", "MapWidth",
                   "MapHeight", "GmapType", "EleLow", "EleDiv", "EleCss", "TireSec", "KeepSec"]
MISSILE_ATTRS = ("pos", "pos2", "blood", "color", "car", "sons", "son", "space", "radar",
                 "stop", "time", "ste", "see", "e2", "att", "aim", "tar", "task", "data")
CARRIER_ATTRS = ("pos", "pos2", "sons", "stop", "radar", "time", "space", "see", "att",
                 "aim", "tar", "task", "blood", "color")

LOG: list[str] = []
ACTIONS: list[dict] = []


def w(line: str = "") -> None:
    LOG.append(line)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def cookie(sess, name: str) -> str:
    """同名 cookie 可能同时存在于 `/` 与 `/WARGAMEAI<N>`（不同 path）→ `.get()` 会报冲突。"""
    vals = [c.value for c in sess.cookies if c.name == name]
    return vals[-1] if vals else ""


def pick(units: dict, oid: str, keys) -> dict:
    a = units.get(oid) or {}
    return {k: a.get(k, "") for k in keys}


def is_missile(a: dict) -> bool:
    return (a.get("ico") in MISSILE_ICOS or str(a.get("data") or "") in MISSILE_DATA
            or a.get("name") == "巡飞弹")


def roster_carriers(stat: dict, msl: str):
    """按平台自己的**载员编制表**（载车 `son` 逗号表）选巡飞弹的法定载车。

    实测（2026-10-16）：同格有 ZBD100 火力支援车（`space=1`、`son='PLA2-191,VUT5,UGVS,
    RDOG,YY4,WS43'`）与其自己的 UGVS（`space=0`、`son=''`）；把 WS43 装到 UGVS 上
    `com/ObjInto.asp` 回 **False**，而编制表里写着 WS43 的才是它的载车。
    """
    a = stat.get(msl) or {}
    ico = str(a.get("ico") or "")
    # 已上车的弹 `pos=0`，平台把载具格写在 `pos2`（这正是“车内记录”的判定特征）
    hex_ = str(a.get("pos"))
    if hex_ in ("0", "", "None"):
        hex_ = str(a.get("pos2"))
    same = [o for o, b in stat.items() if str(b.get("pos")) == hex_ and o != msl]
    ros = sorted(o for o in same
                 if ico and ico in [x.strip() for x in (stat[o].get("son") or "").split(",")])
    cars = sorted(o for o in same if str(stat[o].get("typeid")) == "2")
    return ros, cars


def com_post(sess, base7: str, objid: str, color: str, command: dict, task: str,
             pos_hex: str) -> dict:
    """平台两段式指令的第 1 段：`com/ComPost.asp` 登记 + `com/ComAjax.asp` 取回 ComID。

    ★ 原始请求参数与响应原文**全部**记录 —— 这就是“监听”要的料。
    """
    params = {"ObjID": objid, "Command": json.dumps(command, ensure_ascii=False),
              "ObjColor": color, "TaskName": task, "ObjPos": pos_hex, "MapID": ""}
    url = base7 + "com/ComPost.asp"
    r = sess.get(url, params=params, timeout=30, headers={"Referer": base7 + "main.html"})
    body = OC.decode(r.content)
    ra = sess.get(base7 + "com/ComAjax.asp", timeout=30,
                  headers={"Referer": base7 + "main.html"})
    echo = OC.decode(ra.content)
    m = re.search(r"(\d+)\|[^|]*\|", echo)
    rec = {"kind": "com_post", "task": task, "at": now(), "step": 1,
           "request": {"url": url, "params": params}, "status": r.status_code,
           "response": body[:400], "comajax": echo[:600],
           "com_id": m.group(1) if m else ""}
    ACTIONS.append(rec)
    w("- [%s] 第1段 ComPost `%s` Command=`%s` → status=%s body=%r；ComAjax=%r（ComID=%s）"
      % (task, objid, json.dumps(command, ensure_ascii=False), r.status_code,
         body[:120], echo[:160], rec["com_id"]))
    return rec


def com_step2(sess, base7: str, path: str, params: dict, task: str) -> dict:
    url = base7 + path
    r = sess.get(url, params=params, timeout=30, headers={"Referer": base7 + "main.html"})
    body = OC.decode(r.content)
    rec = {"kind": "com_step2", "task": task, "at": now(), "step": 2,
           "request": {"url": url, "params": params}, "status": r.status_code,
           "response": body[:400]}
    ACTIONS.append(rec)
    w("- [%s] 第2段 GET `%s` %s → status=%s body=%r"
      % (task, path, params, r.status_code, body[:160]))
    return rec


def snapshot(sess, base7: str, tag: str, msl: str, car: str) -> dict:
    """一份“态势切片”：会话字段 + 弹/载车逐字段 + ObjAjax 动态段。"""
    head = {"Referer": base7 + "main.html"}
    parts = OC.decode(sess.get(base7 + "User_Get.asp", params={"userAgent": "missile"},
                               timeout=30, headers=head).content).split("$")
    stat = OC.parse_obj_get(OC.decode(sess.get(base7 + "Obj_Get.asp", timeout=30,
                                              headers=head).content))
    dyn_raw = OC.decode(sess.get(base7 + "ObjAjax.asp", timeout=30, headers=head).content)
    try:
        dyn = json.loads(dyn_raw.strip().lstrip("\ufeff")) if dyn_raw.strip() else {}
    except Exception:                                            # noqa: BLE001
        dyn = {}
    snap = {"tag": tag, "at": now(), "session": {USER_GET_FIELDS[i]: parts[i]
                                                 for i in range(min(len(parts), 30))},
            "missile_exists": msl in stat, "carrier_exists": car in stat,
            "missile": pick(stat, msl, MISSILE_ATTRS), "carrier": pick(stat, car, CARRIER_ATTRS),
            "missile_in_ajax": dyn.get(msl) if isinstance(dyn, dict) else None,
            "ajax_units": len(dyn) if isinstance(dyn, dict) else 0,
            "units_total": len(stat)}
    if not snap["missile_exists"]:
        snap["missile"] = {}
    if not snap["carrier_exists"]:
        snap["carrier"] = {}
    return snap


def brief(snap: dict) -> str:
    m, c = snap["missile"], snap["carrier"]
    return ("room_second=%s 弹{pos=%s pos2=%s blood=%s color=%r space=%s radar=%r car=%s "
            "stop=%s} 车{sons=%s stop=%s radar=%r space=%s}"
            % (snap["session"].get("RoomSecond"), m.get("pos"), m.get("pos2"),
               m.get("blood"), m.get("color"), m.get("space"), m.get("radar"),
               m.get("car"), m.get("stop"), c.get("sons"), c.get("stop"),
               c.get("radar"), c.get("space")))


def write_report(outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    md = os.path.join(outdir, name or "report.md")
    io.open(md, "w", encoding="utf-8", newline="\n").write("\n".join(LOG) + "\n")
    io.open(os.path.join(outdir, "actions.jsonl"), "w", encoding="utf-8",
            newline="\n").write("".join(json.dumps(a, ensure_ascii=False) + "\n"
                                        for a in ACTIONS))
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="实服巡飞弹部署/发射探针（默认只读）")
    ap.add_argument("--host", default=OC.cred("HEXWAR_BASE") or "http://www.hexwar.cn")
    ap.add_argument("--room", type=int, default=2)
    ap.add_argument("--as", dest="account", choices=("a", "b"), default="b")
    ap.add_argument("--enter", action="store_true", help="进房（改状态）")
    ap.add_argument("--reset-room", action="store_true",
                    help="先初始化房间（清掉陈旧局/释放账号绑定；需再给 --confirm）")
    ap.add_argument("--confirm", action="store_true", help="确认执行 --reset-room")
    ap.add_argument("--start", action="store_true", help="开局（改状态）")
    ap.add_argument("--ai", action="store_true",
                    help="开局用 AIStart=true（无人方交给 AI；实测不传时时钟被“推演暂停”卡在 0）")
    ap.add_argument("--window-test", action="store_true",
                    help="发完一轮后，在“已过第 0 秒”的当前时刻再试一次部署（手册 4.10(3) 窗口）")
    ap.add_argument("--resume", action="store_true",
                    help="时钟被“推演暂停”钉住时，发 `{COM:{pause:0}}` 让它继续走")
    ap.add_argument("--seat", default="",
                    help="以玩家身份入座（RED/BLUE）—— 无人方 + 无 AI 时服务端会暂停推演")
    ap.add_argument("--load", action="store_true", help="部署：把弹装上同格载车（改状态）")
    ap.add_argument("--launch", action="store_true", help="发射（改状态）")
    ap.add_argument("--kill-carrier", action="store_true",
                    help="发射后再打掉发射车，看弹是否消失（改状态，默认关）")
    ap.add_argument("--poll-seconds", type=float, default=240.0, help="发射后监听的真实秒数")
    ap.add_argument("--poll-every", type=float, default=8.0, help="轮询间隔（真实秒）")
    ap.add_argument("--unit", default="", help="指定巡飞弹 Obj id（默认按 ico/data 自动找）")
    ap.add_argument("--carrier", default="", help="指定载车 Obj id")
    ap.add_argument("-o", "--out", default=None, help="报告文件名")
    ns = ap.parse_args(argv)

    outdir = os.path.join(OUTROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    try:
        return _run(ns, outdir)
    except Exception:                                            # noqa: BLE001
        import traceback
        w()
        w("## !! 探针中断")
        w()
        w("```")
        w(traceback.format_exc())
        w("```")
        raise
    finally:
        md = write_report(outdir, ns.out)
        print("报告：%s" % md)


def _run(ns, outdir) -> int:
    OC.OUT = LOG          # ★ 把 `hex_opening_check` 的日志接到本报告里（否则进房/复位过程看不到）
    host = ns.host.rstrip("/")
    base = host + "/WARGAMEAI/"
    ua, pa = ((OC.cred("HEXWAR_USER2"), OC.cred("HEXWAR_PASS2")) if ns.account == "b"
              else (OC.cred("HEXWAR_USER"), OC.cred("HEXWAR_PASS")))
    if not ua or not pa:
        sys.exit("请先设置账号环境变量（HEXWAR_USER/HEXWAR_PASS 或 HEXWAR_USER2/HEXWAR_PASS2）")

    w("# 实服巡飞弹部署/发射探针")
    w()
    w("- 主机 `%s`　推演室 %d（RoomID=%d）　账号 %s　时间 %s"
      % (host, ns.room, OC.ROOM_ID_BASE + ns.room - 1,
         "B" if ns.account == "b" else "A", now()))
    w("- 会改状态的开关：%s" % (", ".join([x for x, on in (("--enter", ns.enter),
                                                            ("--start", ns.start),
                                                            ("--load", ns.load),
                                                            ("--launch", ns.launch),
                                                            ("--kill-carrier", ns.kill_carrier))
                                          if on]) or "（无 → 只读）"))
    w()

    sess = OC.new_session()
    uid = OC.login(sess, base, ua, pa)
    w("- 登录：UserID=`%s` 端口=`%s`" % (uid, cookie(sess, "PortID")))

    if ns.reset_room:
        if not ns.confirm:
            w("- ⚠ 指定了 `--reset-room` 但没有 `--confirm`：**不执行**。")
        else:
            OC.reset_room(sess, host, ns.room)
            time.sleep(1)
            sess = OC.new_session()
            uid = OC.login(sess, base, ua, pa)
            w("- 已初始化推演室 %d，并重新登录（UserID=`%s`，端口=`%s`）"
              % (ns.room, uid, cookie(sess, "PortID")))

    if True:                     # 总是先读一次大厅：拿到该房状态码（只读）
        html = OC.lobby_text(sess, base, host)
        rooms = OC.parse_lobby(html)
        me = [r for r in rooms if r["room"] == ns.room]
        if me:
            r = me[0]
            w("- 大厅里推演室 %d：**%s**(%d)　局内=%s　练习=%s　标题=%s"
              % (r["room"], r["status_text"], r["status"], r["minute"] or "—",
                 r["practice"], r["title"]))
            if r["status"] == 1:
                w("  ⚠ 该房**正在推演**（可能是别人的局）：不加 `--enter` 就停在这里。")
        else:
            w("- ⚠ 大厅里没解析到推演室 %d（房间列表长度 %d）" % (ns.room, len(rooms)))
        if not (ns.enter or ns.load or ns.launch):
            w()
            w("（只读模式：未进房。要实测请加 `--enter --start --load --launch`。）")
            return 0

    base7 = "%s/WARGAMEAI%d/" % (host, ns.room)
    if ns.enter:
        base7 = OC.enter_room(sess, host, ns.room, "NONE", uid, game_name=ua)
        w("- 进房后端口=`%s`" % cookie(sess, "PortID"))
    if ns.start:
        # ★ 刚 SetUpSave 进房后立刻开局，服务端会**短暂**回 `-1`（实测：隔几分钟重试即成功），
        #   所以这里重试几次；`RoomStart` 本身是幂等的（已开局再调仍返回同一个 StartTime）。
        ok = ""
        for attempt in range(1, 7):
            r = sess.get(base7 + "RoomStart.asp",
                         params={"AIStart": "true" if ns.ai else "false", "RoomSecond": "0"},
                         timeout=30, headers={"Referer": base7 + "main.html"})
            body = OC.decode(r.content).strip()
            w("- [%d/6] GET RoomStart.asp?AIStart=%s&RoomSecond=0 -> %r"
              % (attempt, "true" if ns.ai else "false", body[:120]))
            if body not in ("-1", "0", ""):
                ok = body
                break
            time.sleep(5)
        if not ok:
            w("  ⚠ 开局一直失败；若房间是刚初始化的，稍等再试或先 `--load`/换房。")
        time.sleep(2)
        # 时钟是不是真的在走？（实测：无人方无 AI 时服务端会发“推演暂停”、RoomSecond 恒 0）
        c1 = snapshot(sess, base7, "时钟1", "", "")["session"].get("RoomSecond")
        time.sleep(6)
        c2 = snapshot(sess, base7, "时钟2", "", "")["session"].get("RoomSecond")
        w("- 时钟推进检查：RoomSecond %s -> %s（%s）"
          % (c1, c2, "在走" if str(c1) != str(c2) else "★未走（推演暂停）"))
        if str(c1) == str(c2) and ns.seat:
            w("- 试着以玩家身份入座（GameColor=%s）：无人方 + 无 AI 时服务端会暂停推演" % ns.seat)
            r = sess.post(base + "SetUp/SetUpSave.asp", timeout=30, allow_redirects=False, data={
                "UserID": uid, "RoomID": str(OC.ROOM_ID_BASE + ns.room - 1), "GameName": ua,
                "GameColor": ns.seat, "PortID": str(ns.room), "submit": "",
                "UserRoom": str(OC.ROOM_ID_BASE + ns.room - 1)})
            w("  POST SetUp/SetUpSave.asp(color=%s) -> %s Location=%s body=%r"
              % (ns.seat, r.status_code, r.headers.get("Location", ""),
                 OC.decode(r.content)[:160]))
            time.sleep(3)
            c3 = snapshot(sess, base7, "时钟3", "", "")["session"].get("RoomSecond")
            time.sleep(6)
            c4 = snapshot(sess, base7, "时钟4", "", "")["session"].get("RoomSecond")
            w("  入座后时钟：%s -> %s（%s）"
              % (c3, c4, "在走" if str(c3) != str(c4) else "仍未走"))
        if str(c1) == str(c2) and ns.resume:
            w("- 试着恢复推演：`{\"COM\": {\"pause\": 0}}`")
            for objid in ("COM", "0"):
                com_post(sess, base7, objid, "", {"COM": {"pause": 0}}, "推演恢复", "")
                time.sleep(2)
                c5 = snapshot(sess, base7, "时钟5", "", "")["session"].get("RoomSecond")
                time.sleep(6)
                c6 = snapshot(sess, base7, "时钟6", "", "")["session"].get("RoomSecond")
                w("  （ObjID=%s）恢复后时钟：%s -> %s（%s）"
                  % (objid, c5, c6, "在走" if str(c5) != str(c6) else "仍未走"))
                if str(c5) != str(c6):
                    break

    s0 = snapshot(sess, base7, "初始", ns.unit, ns.carrier)
    io.open(os.path.join(outdir, "s0_initial.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(s0, ensure_ascii=False, indent=1))
    if not s0["session"].get("StartTime", "").strip():
        w()
        w("⚠ `StartTime` 为空：推演未开始（先 `--start`）。此处停下。")
        w("- User_Get 前 21 字段：%s" % json.dumps(
            {k: v for k, v in list(s0["session"].items())[:21]}, ensure_ascii=False))
        return 0
    w("- 开局时间 StartTime=%s　RoomSecond=%s　RoomSpeed=%s"
      % (s0["session"].get("StartTime"), s0["session"].get("RoomSecond"),
         s0["session"].get("RoomSpeed")))

    # ---- 选弹与载车 ----
    stat = OC.parse_obj_get(OC.decode(sess.get(base7 + "Obj_Get.asp", timeout=30,
                                               headers={"Referer": base7 + "main.html"}).content))
    msl = ns.unit
    if not msl:
        cands = sorted(o for o, a in stat.items()
                       if is_missile(a) and str(a.get("pos") or "0") != "0")
        if not cands:            # 都已在车内：挑一辆编制表里有弹的车
            cands = sorted(o for o, a in stat.items() if is_missile(a))
        if not cands:
            w("⚠ 图上找不到在场的巡飞弹（ico %s / data %s）" % (MISSILE_ICOS, MISSILE_DATA))
            return 1
        msl = cands[0]
    mhex = stat.get(msl, {}).get("pos")
    mcolor = stat.get(msl, {}).get("color")
    ros, cars = roster_carriers(stat, msl)
    car = ns.carrier
    if not car:
        if ros:
            car = ros[0]
            w("- 选车规则：载车 `son` 编制表里含 `%s` 的车（%s）" % (stat[msl].get("ico"), ros))
        elif cars:
            car = cars[0]
            w("  ⚠ 编制表里没找到含 `%s` 的车，退而用同格第一辆车 %s"
              % (stat[msl].get("ico"), car))
        else:
            w("⚠ 巡飞弹 %s 所在格 %s 没有车辆，找不到载车" % (msl, mhex))
            return 1
    w("- 选定：巡飞弹 `%s`（%s %s data=%s 色=%s 格=%s）；载车 `%s`（%s %s typeid=%s sons=%s）"
      % (msl, stat[msl].get("army"), stat[msl].get("ico"), stat[msl].get("data"), mcolor,
         mhex, car, stat.get(car, {}).get("army"), stat.get(car, {}).get("ico"),
         stat.get(car, {}).get("typeid"), stat.get(car, {}).get("sons")))
    if not (ns.load or ns.launch or ns.kill_carrier):
        w()
        w("（只读模式到此为止：已给出会用的指令，未发送。）")
        return 0

    # ---- 1) 部署（上车）----
    if ns.load:
        w()
        w("## 1 部署（上车）：`{\"%s\": {\"inside\": \"%s\"}}`" % (msl, car))
        com_post(sess, base7, msl, mcolor, {msl: {"inside": car}}, "上车", str(mhex))
        st = ACTIONS[-1]
        com_step2(sess, base7, "com/ObjInto.asp",
                  {"ObjID": msl, "CarID": car, "ComID": st["com_id"]}, "上车")
        time.sleep(2)
        s1 = snapshot(sess, base7, "上车后", msl, car)
        io.open(os.path.join(outdir, "s1_loaded.json"), "w", encoding="utf-8",
                newline="\n").write(json.dumps(s1, ensure_ascii=False, indent=1))
        w("- 上车后：%s" % brief(s1))
        w("- 判据：弹 `pos=0/blood=0/color=''`、载车 `sons=1` ⇒ 上车成功")

    # ---- 2) 发射 ----
    loiter_expect = ""
    try:
        import engine.game as G
        spd = int(s0["session"].get("RoomSpeed") or 3)
        g = G.Game(room_speed=spd)
        loiter_expect = "%s 秒（RoomSpeed=%d × 客户端公式 parseInt(1200/RoomSpeed)）" % (
            g.loiter_sec, spd)
    except Exception as exc:                                     # noqa: BLE001
        loiter_expect = "（引擎公式取不到：%s）" % exc
    if ns.launch:
        w()
        w("## 2 发射：`{\"%s\": {\"off\": 3}}` + `com/Obj2_Out.asp`" % car)
        w("- 引擎侧期望的巡飞上限：%s" % loiter_expect)
        com_post(sess, base7, car, stat[car].get("color"), {car: {"off": 3}}, "发射",
                 str(stat[car].get("pos") or mhex))
        st = ACTIONS[-1]
        t1 = stat[msl].get("time") or ""
        for label, params in (
                ("Time1=弹 time", {"ObjID": car, "ObjTypeID": "3", "Time1": t1,
                                   "ComID": st["com_id"], "ObjStack": "0"}),
                ("Time1=空", {"ObjID": car, "ObjTypeID": "3", "Time1": "",
                              "ComID": st["com_id"], "ObjStack": "0"})):
            rec = com_step2(sess, base7, "com/Obj2_Out.asp", params, "发射(%s)" % label)
            if rec["response"].strip() not in ("0", "", "False"):
                break
            w("  ⚠ 返回 %r，换下一种参数再试" % rec["response"].strip())
        time.sleep(2)
        s2 = snapshot(sess, base7, "发射后", msl, car)
        io.open(os.path.join(outdir, "s2_launched.json"), "w", encoding="utf-8",
                newline="\n").write(json.dumps(s2, ensure_ascii=False, indent=1))
        w("- **发射后立即**：%s" % brief(s2))
        if s2["missile_in_ajax"]:
            w("- ObjAjax 里这枚弹：%s"
              % json.dumps(s2["missile_in_ajax"], ensure_ascii=False)[:300])

        # ---- 3) 监听 ----
        w()
        w("## 3 监听（每 %.0f 真实秒读一次，共 %.0f 秒）" % (ns.poll_every, ns.poll_seconds))
        w()
        w("| # | 本地时刻 | RoomSecond | 弹 pos | pos2 | blood | space | radar | 车 sons | 车 stop | Ajax |")
        w("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
        timeline = io.open(os.path.join(outdir, "timeline.jsonl"), "w", encoding="utf-8",
                           newline="\n")
        t0 = time.time()
        span_true = 8
        last = s2
        while True:
            elapsed = time.time() - t0
            if elapsed > ns.poll_seconds:
                break
            time.sleep(min(ns.poll_every, max(0.0, ns.poll_seconds - elapsed)))
            try:
                sp = snapshot(sess, base7, "poll", msl, car)
            except Exception as exc:                              # noqa: BLE001
                w("| – | %s | 读取失败：%s |" % (now(), exc))
                continue
            timeline.write(json.dumps(sp, ensure_ascii=False) + "\n")
            timeline.flush()
            last = sp
            i = int(round((time.time() - t0) / ns.poll_every)) or 1
            w("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (i, sp["at"], sp["session"].get("RoomSecond"), sp["missile"].get("pos"),
                 sp["missile"].get("pos2"), sp["missile"].get("blood"),
                 sp["missile"].get("space"), sp["missile"].get("radar"),
                 sp["carrier"].get("sons"), sp["carrier"].get("stop"),
                 1 if sp["missile_in_ajax"] else 0))
            sys.stdout.write("[poll %2d] room_second=%s msl_pos=%s space=%s radar=%s\n"
                             % (i, sp["session"].get("RoomSecond"),
                                sp["missile"].get("pos"), sp["missile"].get("space"),
                                sp["missile"].get("radar")))
            sys.stdout.flush()
            if not sp["missile_exists"]:
                w()
                w("★ **弹从 `Obj_Get` 里消失**：RoomSecond=%s（发射后约 %s 秒真实时间）"
                  % (sp["session"].get("RoomSecond"), int(elapsed)))
                break
            try:
                span_true = int(sp["session"].get("RoomSecond") or 0) - int(
                    s2["session"].get("RoomSecond") or 0)
            except ValueError:
                span_true = 0
        timeline.close()
        w()
        if span_true:
            w("- 监听结束：发射 → 最后一次读到的巡飞跨度 = **sim %s 秒**" % span_true)
        else:
            w("- 监听结束：整段监听里弹**一直在**（服务端没有自己销毁）")
        w("- 最终态：%s" % brief(last))
        io.open(os.path.join(outdir, "s3_final.json"), "w", encoding="utf-8",
                newline="\n").write(json.dumps(last, ensure_ascii=False, indent=1))

    if ns.kill_carrier:
        w()
        w("## 4 打掉发射车（手册 4.10(5)）—— 需要火炮/直瞄参数，本探针只提示未实现")

    if ns.window_test and ns.load:
        w()
        w("## 5 部署窗口测试（手册 4.10(3)：只在“对抗开始前”部署）")
        w("- 现在 RoomSecond=%s（已过第 0 秒）"
          % snapshot(sess, base7, "窗口", "", "")["session"].get("RoomSecond"))
        others = sorted(o for o, a in stat.items()
                        if is_missile(a) and str(a.get("pos") or "0") != "0"
                        and o != msl and str(a.get("color")) == str(mcolor))
        if not others:
            w("- 没有另一枚在场的弹可试")
        else:
            m2 = others[0]
            ros2, _ = roster_carriers(stat, m2)
            car2 = ros2[0] if ros2 else ""
            w("- 试第二枚：弹 `%s`（%s）→ 载车 `%s`"
              % (m2, stat[m2].get("army"), car2 or "（编制表里没有）"))
            if car2:
                com_post(sess, base7, m2, stat[m2].get("color"),
                         {m2: {"inside": car2}}, "上车(窗口测试)", str(stat[m2].get("pos")))
                st2 = ACTIONS[-1]
                rec = com_step2(sess, base7, "com/ObjInto.asp",
                                {"ObjID": m2, "CarID": car2, "ComID": st2["com_id"]},
                                "上车(窗口测试)")
                time.sleep(2)
                s5 = snapshot(sess, base7, "窗口测试后", m2, car2)
                w("- 结果：弹 pos=%s blood=%s color=%r；载车 sons=%s ⇒ %s"
                  % (s5["missile"].get("pos"), s5["missile"].get("blood"),
                     s5["missile"].get("color"), s5["carrier"].get("sons"),
                     "平台**允许**过了第 0 秒再部署" if rec["response"].strip() == "True"
                     else "平台**拒绝**（响应 %r）" % rec["response"].strip()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
