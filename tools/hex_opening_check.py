# -*- coding: utf-8 -*-
"""到**实际服务器**核对想定 7527 的**开局初设**（载员到底在不在车上）。

要回答的问题（本仓库与用户曾有分歧）：

    “初设（想定加载后的开局态）里，步兵班/无人车/无人机/巡飞弹
      到底是**在图上独立在场**（`sons=0`、`car=Obj_0`），还是**已经装在车里**？”

做法：登录 → （可选）复位房间 → 进房 → 读 `Obj_Get.asp` / `ObjAjax.asp` / `Report_All.asp`，
把逐单位 `pos / pos2 / blood / color / car / sons / son` 与**随仓库发布的**
`scenario/7527/units.json` 逐条比对，输出 JSON + Markdown 报告。

## 实测结论（2026-10-16，练习服 `www.hexwar.cn`）

在**空闲**推演室（大厅里状态 0，见 `--list-rooms`）用导演账号实测：

- 进房后 `StartTime` 为空：静态单位 **36**、在场 `36`、**车内 0**、载具 `sons>0` **0**，
  且 **36 个单位的 `pos2` 全等于自己的 `pos`**；
- `RoomStart.asp` 开局后（第 0 秒）再读：结果**完全相同** —— 平台**不会**在开局瞬间
  把乘员/巡飞弹自动装上车；
- 中局对照（推演室7，开局 11 小时后）：38 单位里 18 条已上车，
  车内记录的形态是 `pos=0 / pos2=<载具格> / blood=0 / color='' / car=Obj_0`，
  载具 `sons=N` —— 与 `engine/game.py` 的载员模型一致。

⇒ “初设已经全部部署上车”**不成立**；发布数据与实服对比 **0 差异**
（`pos2` 残留的 6 处已由 `tools/fix_opening_pos2.py` 按同样证据修正）。

## 安全约定

- 凭据**只从环境变量读**，绝不写进代码、绝不打印：
  `HEXWAR_USER` / `HEXWAR_PASS`（A 号，当导演）；`HEXWAR_USER2` / `HEXWAR_PASS2`（B 号，开局要用）。
  主机用 `--host` 或 `HEXWAR_BASE`（默认练习服 `http://www.hexwar.cn`）。
- **默认只读**：不带任何开关时只登录 + 读大厅房间列表，不进房、不下令、不修改任何状态。
- 会改状态的动作必须显式开关：`--enter`（进房）、`--reset`（复位/初始化房间，需再加 `--confirm`）、
  `--start`（开始推演，需要两个账号）。
- 报告写在 `analysis/`（**gitignore**，不入库、不发布）。

用法::

    # 0) 只读：大厅房间列表与状态码（0 可进入 / 1 正在推演 / 2 需先初始化）
    py -3.13 tools\\hex_opening_check.py --list-rooms

    # 1) 进房（导演）后读“开局前”的初设
    py -3.13 tools\\hex_opening_check.py --room 7 --enter

    # 2) 完整复测：复位 -> 导演进房 -> 读开局前 -> 开局 -> 读开局后
    py -3.13 tools\\hex_opening_check.py --room 7 \\
        --reset --confirm --enter --start

    # 3) 用 B 号账号（HEXWAR_USER2）在**空闲**房（状态 0）复测，不动别人在局的房：
    py -3.13 tools\\hex_opening_check.py --as b --room 2 --enter --start

    # 4) 想对比赛服做同样的事，加 --host http://43.143.209.228
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO_UNITS = os.path.join(ROOT, "scenario", "7527", "units.json")
OUTDIR = os.path.join(ROOT, "analysis")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
SCENARIO_ID = "7527"
ROOM_ID_BASE = 222          # RoomID = 222 + (推演室序号 - 1)

#: 大厅房间卡片里 `ColorList(RoomID, 想定, 标题, 序号, 状态)` 第 5 个参数的含义
#: （与官方大厅点击行为一致：状态 2 的房间会先弹“是否初始化该推演室？”）
ROOM_STATUS = {0: "空闲可进", 1: "正在推演", 2: "需先初始化"}

OUT: list[str] = []
RESULT: dict = {}


def w(line: str = "") -> None:
    OUT.append(line)


def decode(b: bytes) -> str:
    for enc in ("utf-8", "gb18030"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def cred(name: str) -> str:
    """取凭据：先**进程环境变量**，再 Windows 用户/机器级环境变量（HKCU/HKLM）。

    为什么多这一步：`setx` 只写注册表，**不会**影响已经打开的终端 —— 加了这个回退，
    用户在自己终端里 `setx HEXWAR_USER/HEXWAR_PASS` 一次即可，无需重启 VS Code。
    ★ 绝不打印取值；返回字符串仅用于登录。
    """
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        import winreg
    except ImportError:
        return ""
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, r"Environment") as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            continue
    return ""


def parse_obj_get(html):
    """官方 `Obj_Get.asp`：每个单位是一个 `<div id="Obj_xxxx" ...>`。"""
    from bs4 import BeautifulSoup
    units = {}
    for d in BeautifulSoup(html, "lxml").find_all("div"):
        oid = d.get("id", "")
        if not oid.startswith("Obj_"):
            continue
        attrs = {k: d.get(k) for k in d.attrs if k != "id"}
        attrs["_id"] = oid
        units[oid] = attrs
    return units


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    return s


def login(sess: requests.Session, base: str, user: str, pwd: str) -> str:
    """官方登录：先 GET 登录页（拿 ASPSESSIONID），再 POST `login/`。返回 UserID。"""
    sess.get(base, timeout=30)
    sess.post(base + "login/", data={"UserName": user, "UserPassword": pwd},
              timeout=30, allow_redirects=True)
    return sess.cookies.get("UserID", "")


def parse_lobby(html: str) -> list:
    """解析大厅房间卡片。

    每间房是一段 `<label>`：RoomID 在 radio 的 `value` 上，其余在 `onclick` 上：

        <input type="radio" name="UserRoom" value="222"
               onclick="ColorList(this.value,'7527','城市攻防作战（2V2）（推演室1）',1,1)" />
                                      ↑想定    ↑标题                        ↑序号 ↑状态

    卡片里另有 `3倍速`、`练习 N`、`<em>第N分钟</em>`（正在推演时才有）。
    """
    rooms = []
    for m in re.finditer(r'<label[^>]*>(.*?)</label>', html, re.S):
        seg = m.group(1)
        rid = re.search(r'name="UserRoom"[^>]*value="(\d+)"', seg)
        call = re.search(r"ColorList\(this\.value,'(\d+)','(.*?)',(\d+),(\d+)\)", seg)
        if not rid or not call:
            continue
        scen, title, room_no, status = call.groups()
        em = re.search(r"<em>(.*?)</em>", seg, re.S)
        stat = re.search(r'<div class="stat">(.*?)</div>', seg, re.S)
        body = stat.group(1) if stat else ""
        practice = re.search(r"<p>练习\s*(\d+)</p>", body)
        counts = dict(re.findall(r'class="text-(\w+)">\S+?\s*(\d+)</span>', body))
        speed = re.search(r"<span>(.*?)</span>", seg, re.S)
        rooms.append({"room": int(room_no), "room_id": int(rid.group(1)), "scenario": scen,
                      "title": title, "status": int(status),
                      "status_text": ROOM_STATUS.get(int(status), "?"),
                      "minute": " ".join(em.group(1).split()) if em else "",
                      "practice": int(practice.group(1)) if practice else None,
                      "counts": counts,
                      "speed": " ".join(speed.group(1).split()) if speed else ""})
    return rooms


def lobby_text(sess: requests.Session, base: str, host: str) -> str:
    """读大厅。★ 登录响应把 `UserID` 写在 `path=/WARGAMEAI<N>`（不是 `/`），
    而大厅在 `/WARGAMEAI/SetUp/` —— 所以要把 `UserID` 临时再挂一份到根路径，
    且**必须带域名**（`CookieJar.set` 不带 domain 时匹配不上，会被回“请先登录”）。
    """
    jar = sess.cookies
    uid = jar.get("UserID", "")
    domain = host.split("//")[-1].split("/")[0].split(":")[0]
    cookie = None
    if uid:
        cookie = requests.cookies.create_cookie(name="UserID", value=uid,
                                                domain=domain, path="/")
        jar.set_cookie(cookie)
    try:
        r = sess.get(base + "SetUp/SetUp_Transfer/", timeout=40, allow_redirects=True)
        body = decode(r.content)
        os.makedirs(OUTDIR, exist_ok=True)
        with io.open(os.path.join(OUTDIR, "opening_check_lobby.html"), "w",
                     encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        return body
    finally:
        if cookie is not None:
            try:
                jar.clear(domain, "/", "UserID")
            except KeyError:
                pass


def enter_room(sess: requests.Session, host: str, room: int, color: str, uid: str,
               game_name: str = ""):
    """官方进房：POST SetUp/SetUpSave.asp → 302 → /WARGAMEAI<room>/Login/Login_Transfer/。

    载荷与仓库里已验证过的 `tools/hex_transport_capture.py::enter()` 保持一致
    （`GameName` 填账号名、`submit` 为空串、`RoomID`/`UserRoom` 是 **222+序号-1**、`PortID` 是序号）。
    """
    base = host + "/WARGAMEAI/"
    room_db_id = ROOM_ID_BASE + room - 1
    sess.headers["Referer"] = base + "SetUp/"
    r = sess.post(base + "SetUp/SetUpSave.asp", timeout=30, allow_redirects=False, data={
        "UserID": uid, "RoomID": str(room_db_id), "GameName": game_name,
        "GameColor": color, "PortID": str(room), "submit": "",
        "UserRoom": str(room_db_id)})
    loc = r.headers.get("Location", "")
    w("- POST SetUp/SetUpSave.asp(room=%d, RoomID=%d, color=%s) -> %s Location=%s"
      % (room, room_db_id, color, r.status_code, loc))
    if not loc:
        w("  ⚠ 未拿到 302，响应正文：%r" % decode(r.content)[:200])
        base7 = "%s/WARGAMEAI%d/" % (host, room)
        sess.get(base7, timeout=30)
        sess.get(base7 + "Login/Login_Transfer/?PortID=%d" % room, timeout=30)
        return base7
    target = requests.compat.urljoin(base + "SetUp/", loc)
    r2 = sess.get(target, timeout=30, allow_redirects=True)
    m = re.search(r"WARGAMEAI(\d+)/", loc) or re.search(r"WARGAMEAI(\d+)/", r2.url)
    base7 = "%s/WARGAMEAI%s/" % (host, m.group(1)) if m else "%s/WARGAMEAI%d/" % (host, room)
    w("- 客户端目录 `%s`（跟随跳转 -> %s len=%d）" % (base7, r2.url, len(r2.content)))
    return base7


def reset_room(sess: requests.Session, host: str, room: int) -> bool:
    """官方“初始化推演室”：GET 拿自动提交表单，再 POST 它。"""
    url = "%s/WARGAMEAI%d/Reset/Post.asp" % (host, room)
    r = sess.get(url, params={"scenarioId": SCENARIO_ID, "roomId": str(room)}, timeout=30)
    raw = decode(r.content)
    m = re.search(r'<form[^>]*action="([^"]*)"[^>]*>(.*?)</form>', raw, re.S)
    w("- Reset GET -> %s len=%d" % (r.status_code, len(r.content)))
    if not m:
        w("  ⚠ 未拿到自动提交表单：%r" % raw[:200])
        return False
    data = {}
    for im in re.finditer(r'<textarea[^>]*name="([^"]+)"[^>]*>(.*?)</textarea>', m.group(2), re.S):
        data[im.group(1)] = im.group(2).strip()
    r2 = sess.post(requests.compat.urljoin(url, m.group(1)), data=data, timeout=30)
    w("- Reset POST -> %s len=%d" % (r2.status_code, len(r2.content)))
    return True


def read_state(sess: requests.Session, base7: str, tag: str):
    """读一份状态：会话参数 + 静态单位 + 动态单位 + 计分。"""
    head = {"Referer": base7 + "main.html"}
    ug = decode(sess.get(base7 + "User_Get.asp", params={"userAgent": "opening"},
                         timeout=30, headers=head).content)
    parts = ug.split("$")
    if len(parts) < 21:
        w("- [%s] User_Get 未进房/不可读：%r" % (tag, parts[:4]))
        return None
    static = parse_obj_get(decode(sess.get(base7 + "Obj_Get.asp", timeout=30,
                                           headers=head).content))
    dyn_raw = decode(sess.get(base7 + "ObjAjax.asp", timeout=30, headers=head).content)
    try:
        dyn = json.loads(dyn_raw.strip().lstrip("\ufeff")) if dyn_raw.strip() else {}
    except Exception:                                          # noqa: BLE001
        dyn = {}
    report = decode(sess.get(base7 + "Report_All.asp", timeout=30, headers=head).content)
    snap = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
            "session_fields": parts[:30], "static_units": static,
            "dynamic_unit_count": len(dyn),
            "report_all": " ".join(report.split())[:1500]}
    onboard = sorted(((o, a) for o, a in static.items()
                      if str(a.get("pos") or "") == "0"
                      or str(a.get("blood") or "") == "0"),
                     key=lambda kv: str(kv[0]))
    carriers = sorted((o, a) for o, a in static.items()
                      if str(a.get("sons") or "0") not in ("0", ""))
    w("- [%s] 静态单位=%d 动态单位=%d　在场(pos>0 且 blood>0)=%d　车内(pos=0 或 blood=0)=%d"
      "　载具 sons>0 = %d　StartTime=%r"
      % (tag, len(static), len(dyn),
         sum(1 for a in static.values() if str(a.get("pos") or "0") != "0"
             and str(a.get("blood") or "0") != "0"),
         len(onboard), len(carriers), parts[3]))
    for o, a in carriers:
        w("    · 载具 %s %s sons=%s color=%s pos=%s son=%r"
          % (o, a.get("ico"), a.get("sons"), a.get("color"), a.get("pos"), a.get("son")))
    for o, a in onboard[:20]:
        w("    · 车内 %s %s pos=%s pos2=%s blood=%s color=%r car=%s"
          % (o, a.get("ico"), a.get("pos"), a.get("pos2"), a.get("blood"),
             a.get("color"), a.get("car")))
    return snap


def compare_with_published(static, tag: str):
    """把实时开局态与随仓库发布的 `scenario/7527/units.json` 逐条比对。"""
    if not os.path.isfile(SCENARIO_UNITS):
        w("- 找不到 %s，跳过比对" % SCENARIO_UNITS)
        return None
    with io.open(SCENARIO_UNITS, encoding="utf-8") as fh:
        published = json.load(fh)
    live_by_army = {"%s|%s" % (a.get("army"), a.get("ico")): a for a in static.values()}
    rows, only_live, only_pub = [], [], []
    pub_by_army = {"%s|%s" % (u.get("army"), u.get("ico")): u for u in published}
    for key, pub in pub_by_army.items():
        live = live_by_army.get(key)
        if live is None:
            only_pub.append(key)
            continue
        diff = {}
        for field in ("pos", "pos2", "blood", "color", "sons", "car", "son"):
            a, b = str(live.get(field) or ""), str(pub.get(field) or "")
            if a != b:
                diff[field] = {"live": a, "published": b}
        rows.append({"army": key, "id_live": live.get("_id"), "diff": diff})
    for key in live_by_army:
        if key not in pub_by_army:
            only_live.append(key)
    changed = [r for r in rows if r["diff"]]
    w("- 与发布初设比对（按 `army|ico` 配对）：匹配 %d 条，其中**有差异 %d 条**；"
      "只在实时/只在发布：%d / %d" % (len(rows), len(changed), len(only_live), len(only_pub)))
    for r in changed:
        w("    · %s：%s" % (r["army"], json.dumps(r["diff"], ensure_ascii=False)))
    RESULT.setdefault("compare", {})[tag] = {
        "matched": len(rows), "changed": changed,
        "only_live": only_live, "only_published": only_pub}
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="到实际服务器核对 7527 开局初设")
    ap.add_argument("--host", default=cred("HEXWAR_BASE") or "http://www.hexwar.cn")
    ap.add_argument("--room", type=int, default=7, help="推演室序号（RoomID = 222+序号-1）")
    ap.add_argument("--list-rooms", action="store_true", help="只读：大厅房间列表")
    ap.add_argument("--enter", action="store_true", help="以导演身份进房（改状态）")
    ap.add_argument("--reset", action="store_true", help="复位/初始化房间（改状态，需 --confirm）")
    ap.add_argument("--confirm", action="store_true", help="确认执行 --reset")
    ap.add_argument("--start", action="store_true", help="由导演开局（需第二账号）")
    ap.add_argument("--as", dest="account", choices=("a", "b"), default="a",
                    help="用哪个账号：a=HEXWAR_USER（默认，主持/导演），b=HEXWAR_USER2")
    ap.add_argument("--color", default="NONE", help="进房身份，默认 NONE=导演")
    ap.add_argument("-o", "--out", default=None, help="报告文件名（写在 analysis/ 下）")
    ns = ap.parse_args(argv)

    host = ns.host.rstrip("/")
    base = host + "/WARGAMEAI/"
    if ns.account == "b":
        uA, pA = cred("HEXWAR_USER2"), cred("HEXWAR_PASS2")
        uB, pB = cred("HEXWAR_USER"), cred("HEXWAR_PASS")
    else:
        uA, pA = cred("HEXWAR_USER"), cred("HEXWAR_PASS")
        uB, pB = cred("HEXWAR_USER2"), cred("HEXWAR_PASS2")
    if not uA or not pA:
        sys.exit("请先设置环境变量 %s / %s（setx 写用户级也可以；不要写进代码/文件）"
                 % (("HEXWAR_USER2", "HEXWAR_PASS2") if ns.account == "b"
                    else ("HEXWAR_USER", "HEXWAR_PASS")))

    w("# 7527 开局初设核对（实际服务器）")
    w()
    w("- 主机：`%s`　推演室：%d（RoomID=%d）　身份：%s　账号：%s"
      % (host, ns.room, ROOM_ID_BASE + ns.room - 1, ns.color,
         "B(HEXWAR_USER2)" if ns.account == "b" else "A(HEXWAR_USER)"))
    w("- 只读项：登录 + 读页面；改状态项：%s"
      % ", ".join([x for x, on in (("--reset", ns.reset and ns.confirm),
                                   ("--enter", ns.enter), ("--start", ns.start)) if on])
      or "（无）")
    w()

    sess = new_session()
    uid = login(sess, base, uA, pA)
    w("- 登录：UserID=`%s` 角色=`%s` 房=`%s` 端口=`%s`"
      % (uid, sess.cookies.get("UserRole", ""), sess.cookies.get("UserRoom", ""),
         sess.cookies.get("PortID", "")))
    RESULT["host"] = host
    RESULT["room"] = ns.room

    if ns.list_rooms or not (ns.enter or ns.reset):
        html = lobby_text(sess, base, host)
        rooms = parse_lobby(html)
        w()
        w("- 大厅页长度 %d（原文存 `analysis/opening_check_lobby.html`），解析到 %d 间房"
          % (len(html), len(rooms)))
        if rooms:
            w()
            w("| 推演室 | RoomID | 想定 | 状态 | 局内 | 倍速 | 练习 | 标题 |")
            w("| ---: | ---: | --- | --- | --- | --- | ---: | --- |")
            for r in rooms:
                w("| %d | %d | %s | %s(%d) | %s | %s | %s | %s |"
                  % (r["room"], r["room_id"], r["scenario"], r["status_text"],
                     r["status"], r["minute"] or "—", r["speed"], r["practice"], r["title"]))
            free = [r["room"] for r in rooms
                    if r["scenario"] == SCENARIO_ID and r["status"] == 0
                    and not r["minute"] and not (r["practice"] or 0)]
            w()
            w("- 想定 `%s` 且**空闲**（状态 0、无局内时间、练习 0）的推演室：%s"
              % (SCENARIO_ID, ", ".join(str(x) for x in free) or "（无）"))
        else:
            clean = re.sub(r"<[^>]+>", " ", html)
            w("    （解析为空，页面文本前 400 字：%s）" % " ".join(clean.split())[:400])
        RESULT["lobby"] = rooms
        if not (ns.enter or ns.reset):
            w()
            w("（默认只读模式：未进房。要读开局初设请加 `--enter`。）")
            write_report(ns.out)
            return 0

    base7 = "%s/WARGAMEAI%d/" % (host, ns.room)
    if ns.reset:
        if not ns.confirm:
            w("⚠ 指定了 `--reset` 但没有 `--confirm`：**不执行**，只读现状。")
        else:
            reset_room(sess, host, ns.room)
            time.sleep(1)
            sess = new_session()
            uid = login(sess, base, uA, pA)
            w("- 复位后重新登录：UserID=`%s`" % uid)
    if ns.enter:
        base7 = enter_room(sess, host, ns.room, ns.color, uid, game_name=uA)

    before = read_state(sess, base7, "开局前")
    if before:
        RESULT["before"] = before
        compare_with_published(before["static_units"], "开局前")

    if ns.start:
        if not uB or not pB:
            w("⚠ `--start` 需要第二个账号：请设置另一个账号的环境变量")
        else:
            sessB = new_session()
            uidB = login(sessB, base, uB, pB)
            w("- 第二账号登录：UserID=`%s`" % uidB)
            try:
                enter_room(sessB, host, ns.room, "RED", uidB, game_name=uB)
            except Exception as exc:                                  # noqa: BLE001
                w("  ⚠ 第二账号进房异常（不阻断开局）：%s" % exc)
            ok = ""
            for params in ({"AIStart": "false", "RoomSecond": "0"},
                           {"AIStart": "true", "RoomSecond": "0"}):
                r = sess.get(base7 + "RoomStart.asp", params=params, timeout=30,
                             headers={"Referer": base7 + "main.html"})
                txt = decode(r.content).strip()
                w("- GET RoomStart.asp %s -> %r" % (params, txt[:120]))
                if txt not in ("-1", "0", ""):
                    ok = txt
                    break
                time.sleep(1)
            time.sleep(2)
            after = read_state(sess, base7, "开局后")
            if after:
                RESULT["after"] = after
                compare_with_published(after["static_units"], "开局后")
            w("- 开局结果：%s" % ("成功" if ok else "**失败**（返回 -1/0/空）"))

    write_report(ns.out)
    return 0


def write_report(name):
    os.makedirs(OUTDIR, exist_ok=True)
    md = os.path.join(OUTDIR, name or "opening_check.md")
    js = os.path.join(OUTDIR, (name or "opening_check.md").rsplit(".", 1)[0] + ".json")
    with io.open(md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(OUT) + "\n")
    with io.open(js, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(RESULT, fh, ensure_ascii=False, indent=1)
    print("已写入 %s\n已写入 %s" % (md, js))
    for line in OUT:
        print(line)


if __name__ == "__main__":
    sys.exit(main())
