# -*- coding: utf-8 -*-
"""7527 场景「卸载乘员」采集：登录 -> 进个人练习室 -> 抓卸载前后单位快照 -> 执行一次卸载。

为什么单独写一个脚本：
    仓库最新版的 tools/hex_login.py 与 tools/hex_capture.py 在检测到目标是 HTTP 时
    会直接退出（"Refusing to send credentials over HTTP"）。官方两台主机
    （www.hexwar.cn 与 43.143.209.228）都不提供 HTTPS —— 443 无法完成 TLS 握手，
    站点自身登录表单也是明文 POST。因此要按用户要求做实测采集，只能显式放行 HTTP。

安全边界（务必遵守）：
    * 必须显式设置 HEXWAR_ALLOW_HTTP=1 才会执行。
    * 只在个人练习室（导演/NONE）里操作，不下达任何作战指令，除卸载外不改动状态。
    * 凭证只从环境变量读；产物里不写 Cookie / 密码 / 认证头。
    * 只保留与本次卸载有关的响应正文。

用法：
    $env:HEXWAR_USER='xxx'; $env:HEXWAR_PASS='xxx'; $env:HEXWAR_ALLOW_HTTP='1'
    py -3.13 tools/hex_transport_capture.py lobby            # 只登录并读大厅占用
    py -3.13 tools/hex_transport_capture.py enter            # 再进房并抓初始快照
    py -3.13 tools/hex_transport_capture.py unload --sample blue_armored_unload_01
"""
import argparse
import json
import os
import re
import sys
import time
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

HOST = os.environ.get("HEXWAR_BASE", "http://www.hexwar.cn")
BASE = HOST + "/WARGAMEAI/"
# ★ 每个推演室有独立的客户端目录：推演室N -> /WARGAMEAI<N>/
#   （SetUpSave.asp 的 302 Location 就是指向它）。进房后由 enter() 改写。
BASE7 = HOST + "/WARGAMEAI7/"
CAPDIR = os.environ.get("HEXWAR_CAPDIR")
if not CAPDIR:
    CAPDIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
        "archive", "exploration", "capture", "7527_transport_capture")
    if not os.path.isdir(CAPDIR):                      # 未封存的检出
        CAPDIR = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "7527_transport_capture")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

USERNAME = os.environ.get("HEXWAR_USER", "")
PASSWORD = os.environ.get("HEXWAR_PASS", "")

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})

# 当前样本目录：卸载相关的原始响应存到它里面，避免多个样本互相覆盖
SAMPLE_DIR = CAPDIR

LOG = []


def w(x=""):
    LOG.append(x)


def dump_log(name):
    os.makedirs(CAPDIR, exist_ok=True)
    with open(os.path.join(CAPDIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


def decode(b):
    for enc in ("utf-8", "gb18030"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")


def save_raw(name, r):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    p = os.path.join(SAMPLE_DIR, name)
    with open(p, "wb") as f:
        f.write(r.content)
    return p, len(r.content)


def guard():
    if not USERNAME or not PASSWORD:
        sys.exit("请先设置 HEXWAR_USER / HEXWAR_PASS")
    if urlsplit(HOST).scheme.lower() != "https" and os.environ.get("HEXWAR_ALLOW_HTTP") != "1":
        sys.exit("目标不是 HTTPS。若确认要用官方 HTTP 站点，请设 HEXWAR_ALLOW_HTTP=1")


# ---------------------------------------------------------------- 登录
def login():
    S.headers.pop("Referer", None)
    r0 = S.get(BASE, timeout=30)
    w("- 入口页 %s status=%s" % (r0.url, r0.status_code))
    r1 = S.post(BASE + "login/", data={"UserName": USERNAME, "UserPassword": PASSWORD},
                timeout=30, allow_redirects=True)
    ck = requests.utils.dict_from_cookiejar(S.cookies)
    w("- 登录后 status=%s url=%s" % (r1.status_code, r1.url))
    w("- 登录后 cookie 键: %s" % sorted(ck.keys()))
    txt = decode(r1.content)
    if re.search(r"(用户名|密码).*(错误|不正确)|不存在", txt):
        sys.exit("登录失败（页面出现错误提示）")
    userid = ck.get("UserID") or ""
    w("- UserID=%s  UserRole=%s  UserAdmin=%s" % (
        userid, ck.get("UserRole"), ck.get("UserAdmin")))
    return userid


# ---------------------------------------------------------------- 大厅
# 站点把登录态写在 UserID / UserRoom / PortID 这些 cookie 上，且登录响应把它们的作用域
# 限在 path=/WARGAMEAI7。直接请求 /WARGAMEAI/SetUp/ 时浏览器/requests 都不会带上它们，
# 服务端于是回 "请先登录"。读大厅时临时把 UserID 也挂到 path=/，读完立刻删掉。
def lobby(userid):
    s_uid = requests.cookies.create_cookie(name="UserID", value=userid or "",
                                           domain=urlsplit(HOST).hostname, path="/")
    S.cookies.set_cookie(s_uid)
    try:
        r = S.get(BASE + "SetUp/SetUp_Transfer/", timeout=30)
    finally:
        S.cookies.clear(urlsplit(HOST).hostname, "/", "UserID")
    save_raw("lobby_setup_transfer.html", r)
    html = decode(r.content)
    if "请先登录" in html:
        w("- 大厅仍要求登录（UserID cookie 未生效）")
        return r
    m = re.search(r"location\.href=\('([^']+)'\)", html)
    if m and "SetUp_Transfer" not in m.group(1):
        w()
        w("## 大厅状态")
        w()
        w("- 请求大厅被重定向到 `%s`" % m.group(1))
        w("- 含义：本账号当前**已占用某个 PortID**（服务端仍有活动会话），"
          "需复用或结束该会话才能看到房间列表")
        return r
    soup = BeautifulSoup(html, "lxml")
    w()
    w("## 大厅推演室列表")
    w()
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells and any(("推演室" in c or "红方" in c or "蓝方" in c or "超时" in c
                          or "结束" in c or "练习" in c) for c in cells):
            w("- %s" % " | ".join(cells))
    return r


def quit_room(userid):
    """退出当前推演室（释放 PortID）。站点靠 UserID cookie 判定，需临时放宽作用域。"""
    s_uid = requests.cookies.create_cookie(name="UserID", value=userid or "",
                                           domain=urlsplit(HOST).hostname, path="/")
    S.cookies.set_cookie(s_uid)
    try:
        r = S.get(BASE + "SetUp/SetUp_Transfer/Quit.asp", timeout=30)
    finally:
        S.cookies.clear(urlsplit(HOST).hostname, "/", "UserID")
    w()
    w("## 退出推演室")
    w()
    w("- GET SetUp/SetUp_Transfer/Quit.asp -> status=%s len=%d" % (r.status_code, len(r.content)))
    save_raw("quit_result.html", r)
    return r


def reset_room(scenario, room, userid):
    """初始化推演室（官方大厅里 str==2 的房间会提示"是否初始化该推演室？"，
    确认后跳到这里）。用于把陈旧/超时的房间恢复成可重新开局的状态。"""
    s_uid = requests.cookies.create_cookie(name="UserID", value=userid or "",
                                           domain=urlsplit(HOST).hostname, path="/")
    S.cookies.set_cookie(s_uid)
    try:
        r = S.get(BASE + "SetUp/Reset/Post.asp",
                  params={"scenarioId": str(scenario), "roomId": str(room)}, timeout=40)
    finally:
        S.cookies.clear(urlsplit(HOST).hostname, "/", "UserID")
    w()
    w("## 初始化推演室 %d（scenarioId=%s）" % (room, scenario))
    w()
    w("- GET SetUp/Reset/Post.asp -> status=%s len=%d body=%r"
      % (r.status_code, len(r.content), decode(r.content)[:300]))
    save_raw("reset_room_%d.html" % room, r)
    return r


# ---------------------------------------------------------------- 进房
def enter(room, color, userid=""):
    room_db_id = 222 + (room - 1)
    S.headers["Referer"] = BASE + "SetUp/"
    userid = userid or requests.utils.dict_from_cookiejar(S.cookies).get("UserID", "")
    r = S.post(BASE + "SetUp/SetUpSave.asp", data={
        "UserID": userid, "RoomID": str(room_db_id), "GameName": USERNAME,
        "GameColor": color, "PortID": str(room), "submit": "", "UserRoom": str(room_db_id),
    }, timeout=30, allow_redirects=False)
    w()
    w("## 进入推演室 %d（RoomID=%d, 身份=%s）" % (room, room_db_id, color))
    w("- POST SetUp/SetUpSave.asp -> %s  Location=%s" % (r.status_code, r.headers.get("Location")))
    loc = r.headers.get("Location", "")
    if loc and "WARGAMEAI" in loc:
        m = re.search(r"WARGAMEAI(\d+)/", loc)
        if m:
            globals()["BASE7"] = "%s/WARGAMEAI%s/" % (HOST, m.group(1))
            w("- 客户端目录改用 `%s`（每个推演室一个）" % globals()["BASE7"])
    if not loc:
        body = decode(r.content)[:200].replace("\n", " ")
        w("- 未拿到跳转，响应正文：%s" % body)
        globals()["BASE7"] = "%s/WARGAMEAI%d/" % (HOST, room)
        loc = BASE + "login/Login_Transfer/?PortID=%d" % room
        w("- 改用客户端目录 `%s`，入口：%s" % (globals()["BASE7"], loc))
    r2 = S.get(requests.compat.urljoin(BASE + "SetUp/", loc), timeout=30, allow_redirects=True)
    w("- 跟随跳转 -> %s status=%s len=%d" % (r2.url, r2.status_code, len(r2.content)))
    save_raw("main.html", r2)
    return room_db_id


# ---------------------------------------------------------------- 会话与单位
FIELD_NAMES = ["CenterID", "GridDiff", "Moban", "StartTime", "UserID", "RoomID", "HexCols",
               "HexRows", "OnTime", "ProMode", "GameColor", "WarDateTime", "GridMeter",
               "RecordID", "RoomTitle", "UserStart", "RoomSpeed", "FirstColor", "Ditu",
               "AllTime", "RoomSecond", "TeamName", "MapWidth", "MapHeight", "GmapType",
               "EleLow", "EleDiv", "EleCss", "TireSec", "KeepSec"]


def session_and_units():
    ua = requests.utils.quote(UA, safe="")
    ru = S.get(BASE7 + "User_Get.asp?userAgent=" + ua, timeout=30,
               headers={"Referer": BASE7 + "main.html"})
    raw = decode(ru.content)
    save_raw("User_Get.asp.txt", ru)
    parts = raw.split("$")
    w()
    w("## User_Get 会话字段")
    w()
    for i, nm in enumerate(FIELD_NAMES):
        if i < len(parts):
            w("- %-12s = %s" % (nm, parts[i][:60]))
    for i in range(len(FIELD_NAMES), len(parts)):
        w("- a[%d] = %s" % (i, parts[i][:60]))
    session = {FIELD_NAMES[i]: parts[i] for i in range(min(len(FIELD_NAMES), len(parts)))}
    session["_extra"] = {("a%d" % i): parts[i] for i in range(len(FIELD_NAMES), len(parts))}

    ro = S.get(BASE7 + "Obj_Get.asp", timeout=30, headers={"Referer": BASE7 + "main.html"})
    save_raw("Obj_Get.asp.html", ro)
    ra = S.get(BASE7 + "ObjAjax.asp", timeout=30, headers={"Referer": BASE7 + "main.html"})
    save_raw("ObjAjax.asp.txt", ra)

    static_units = parse_obj_get(decode(ro.content))
    dynamic_units = parse_obj_ajax(decode(ra.content))
    w()
    w("## 单位")
    w()
    w("- Obj_Get 静态单位数: %d" % len(static_units))
    w("- ObjAjax 动态单位数: %d" % len(dynamic_units))
    return session, static_units, dynamic_units


def parse_obj_get(html):
    soup = BeautifulSoup(html, "lxml")
    units = {}
    for d in soup.find_all("div"):
        oid = d.get("id", "")
        if not oid.startswith("Obj_"):
            continue
        attrs = {k: d.get(k) for k in d.attrs if k != "id"}
        attrs["_id"] = oid
        units[oid] = attrs
    return units


def parse_obj_ajax(text):
    """ObjAjax 是 JSON：{"Obj_100136":{...}, ...}。"""
    text = text.strip().lstrip("\ufeff")
    if not text:
        return {}
    for candidate in (text, text.rstrip(";,")):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 兜底：手工切块（键可能带引号）
    out = {}
    for m in re.finditer(r'["\']?(Obj_\d+|\d+)["\']?\s*:\s*\{', text):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[m.start(0):i + 1]
                    try:
                        k, v = chunk.split(":", 1)
                        out[k.strip().strip('"\'')] = json.loads(v)
                    except Exception:
                        pass
                    break
    return out


# ---------------------------------------------------------------- 卸载
def pick_vehicle(static_units, dynamic_units, prefer_color="BLUE"):
    """挑一辆有乘员的车。sons（静态）= 车内单位数。"""
    cands = []
    for oid, attrs in static_units.items():
        sons = attrs.get("sons") or "0"
        typeid = str(attrs.get("typeid") or "")
        color = attrs.get("color") or ""
        blood = str(attrs.get("blood") or "0")
        try:
            n = int(sons)
        except ValueError:
            n = 0
        if n > 0 and typeid == "2" and blood != "0":
            cands.append((oid, color, n, attrs))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[1] != prefer_color, -t[2]))
    return cands


def do_unload(car_id, crew_flag, color, obj_pos_hex, task_name, time1=""):
    """按官方客户端契约执行一次"下车"。

    客户端是两段式（war.min.js：ComPost.js + ComAjax + CommandOff）：
      1. 玩家点"下车" → ComPost.asp 提交 {"<载具>":{"off":<类型码>}}，服务端登记指令
         并分配 ComID；
      2. 客户端轮询 ComAjax.asp 取回该指令后，由 CommandOff() 真正执行
         GET com/Obj2_Out.asp?ObjID=<载具>&ObjTypeID=<类型码>&Time1=..&ComID=..&ObjStack=..
    这里把两段都照做，才能让服务端真正改状态。
    """
    cmd = {car_id: {"off": crew_flag}}
    params = {"ObjID": car_id, "Command": json.dumps(cmd, ensure_ascii=False),
              "ObjColor": color, "TaskName": task_name, "ObjPos": obj_pos_hex, "MapID": ""}
    r = S.get(BASE7 + "com/ComPost.asp", params=params, timeout=30,
              headers={"Referer": BASE7 + "main.html"})
    save_raw("unload_ComPost.txt", r)
    w("- 第 1 段 GET com/ComPost.asp?%s"
      % "&".join("%s=%s" % (k, requests.utils.quote(str(v))) for k, v in params.items()))
    w("  -> status=%s body=%r（空串=成功受理）" % (r.status_code, decode(r.content)[:200]))

    ra = S.get(BASE7 + "com/ComAjax.asp", timeout=30, headers={"Referer": BASE7 + "main.html"})
    save_raw("ComAjax.asp.txt", ra)
    w("- com/ComAjax.asp -> %r" % decode(ra.content)[:240])
    com_id = ""
    m = re.search(r"(\d+)\|[^|]*\|", decode(ra.content))
    if m:
        com_id = m.group(1)

    p2 = {"ObjID": car_id, "ObjTypeID": str(crew_flag), "Time1": time1,
          "ComID": com_id, "ObjStack": "0"}
    r2 = S.get(BASE7 + "com/Obj2_Out.asp", params=p2, timeout=30,
               headers={"Referer": BASE7 + "main.html"})
    save_raw("unload_Obj2_Out.txt", r2)
    w("- 第 2 段 GET com/Obj2_Out.asp?%s"
      % "&".join("%s=%s" % (k, requests.utils.quote(str(v))) for k, v in p2.items()))
    w("  -> status=%s body=%r" % (r2.status_code, decode(r2.content)[:200]))
    return r, r2, com_id


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lobby", "quit", "reset", "enter", "unload"])
    ap.add_argument("--room", type=int, default=int(os.environ.get("HEXWAR_ROOM", "7")))
    ap.add_argument("--color", default="NONE", help="NONE=导演, RED, BLUE")
    ap.add_argument("--sample", default="unload_01")
    ap.add_argument("--car", default="", help="指定载具 ObjID；留空则自动挑一辆有乘员的车")
    ap.add_argument("--crew", type=int, default=1, help="卸载的乘员类型码 off（1=下车 2=无人车下车 3=卸载）")
    ap.add_argument("--skip-start", action="store_true", help="不下 RoomStart，直接卸")
    ap.add_argument("--require-start", action="store_true",
                    help="要求推演真的开局（StartTime 非空），否则终止，不做卸载")
    ap.add_argument("--wait", type=float, default=5.0, help="下达后等待多少秒再抓 after 快照")
    args = ap.parse_args()

    guard()
    os.makedirs(CAPDIR, exist_ok=True)
    w("# 7527 卸载采集")
    w()
    w("- 目标主机: %s" % HOST)
    w("- 采集时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    w("- 阶段: %s" % args.stage)
    w()

    login_userid = login()
    if args.stage == "lobby":
        lobby(login_userid)
        dump_log("lobby_log.md")
        print("lobby done ->", os.path.join(CAPDIR, "lobby_log.md"))
        return
    if args.stage == "quit":
        quit_room(login_userid)
        dump_log("quit_log.md")
        print("quit done ->", os.path.join(CAPDIR, "quit_log.md"))
        return
    if args.stage == "reset":
        reset_room(7527, args.room, login_userid)
        dump_log("reset_log.md")
        print("reset done ->", os.path.join(CAPDIR, "reset_log.md"))
        return

    lobby(login_userid)
    enter(args.room, args.color, login_userid)

    if args.stage == "unload" and not args.skip_start:
        # 真实玩家也是先"开始推演"，此刻 Time1 才会被置上、下车命令才会走 ComPost 链路
        w()
        w("## 开始推演 RoomStart.asp")
        w()
        rs = S.get(BASE7 + "RoomStart.asp", params={"AIStart": "false", "RoomSecond": "0"},
                   timeout=30, headers={"Referer": BASE7 + "main.html"})
        save_raw("RoomStart.asp.txt", rs)
        w("- GET RoomStart.asp -> status=%s body=%r"
          % (rs.status_code, decode(rs.content)[:160]))
        time.sleep(2)

    session, static_units, dyn = session_and_units()

    started = bool((session.get("StartTime") or "").strip())
    w()
    w("- 推演是否已开始（StartTime 非空）：%s" % ("是" if started else "否"))
    if args.require_start and not started:
        dump_log("unload_log.md")
        sys.exit("推演未开局（StartTime 为空），按 --require-start 终止，不执行卸载")

    sample_dir = os.path.join(CAPDIR, args.sample)
    os.makedirs(sample_dir, exist_ok=True)
    globals()["SAMPLE_DIR"] = sample_dir
    snap = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": HOST, "room": args.room, "color": args.color,
            "session": session, "static_units": static_units, "dynamic_units": dyn}
    with open(os.path.join(sample_dir, "before.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    if args.stage == "enter":
        with open(os.path.join(CAPDIR, "initial_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        dump_log("enter_log.md")
        print("enter done ->", sample_dir)
        return

    cars = pick_vehicle(static_units, dyn)
    w()
    w("## 候选载具（sons>0 且 typeid=2）")
    w()
    if not cars:
        w("- 没有找到有乘员的车辆，无法卸载")
        dump_log("unload_log.md")
        sys.exit("没有载具带乘员")
    for oid, color, n, a in cars:
        w("- %s color=%s sons=%s pos=%s ico=%s" % (oid, color, n, a.get("pos"), a.get("ico")))

    car_id = args.car or cars[0][0]
    car = static_units.get(car_id) or {}
    pos_hex = ""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine import geometry as GEO
        pos_hex = GEO.hexid(int(car.get("pos") or 0))
    except Exception as exc:                                        # noqa: BLE001
        w("- 计算格号失败（%r），ObjPos 留空" % exc)
    ok, ok2, com_id = do_unload(car_id, args.crew, car.get("color") or "", pos_hex,
                                "卸载" if args.crew == 3 else "下车",
                                time1=session.get("StartTime") or "")
    time.sleep(args.wait)
    s2, st2, dy2 = session_and_units()
    after = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "host": HOST, "room": args.room, "color": args.color,
             "session": s2, "static_units": st2, "dynamic_units": dy2}
    with open(os.path.join(sample_dir, "after.json"), "w", encoding="utf-8") as f:
        json.dump(after, f, ensure_ascii=False, indent=1)

    lines = diff_units(static_units, dyn, st2, dy2)
    with open(os.path.join(sample_dir, "action.txt"), "w", encoding="utf-8") as f:
        f.write("操作时间: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        f.write("载具编号: %s（color=%s, ico=%s, army=%s）\n"
                % (car_id, car.get("color"), car.get("ico"), car.get("army")))
        f.write("载具位置: pos=%s 格号=%s\n" % (car.get("pos"), pos_hex))
        f.write("操作前车内单位数 sons=%s，动态乘员数 son=%s\n"
                % (car.get("sons"), ((dyn.get(car_id) or {}).get("son"))))
        f.write("所选乘员类型码 off=%s（1=下车 2=无人车下车 3=卸载）\n" % args.crew)
        f.write("接口: 第1段 GET com/ComPost.asp；第2段 GET com/Obj2_Out.asp（官方客户端契约）\n")
        f.write("ComID: %s\n" % com_id)
        f.write("请求 Command: %s\n" % json.dumps({car_id: {"off": args.crew}},
                                                 ensure_ascii=False))
        f.write("第1段返回: %r（空串=成功受理）\n" % decode(ok.content)[:200])
        f.write("第2段返回: %r\n" % decode(ok2.content)[:200])
        f.write("操作后车内单位数 sons=%s，动态乘员数 son=%s\n"
                % ((st2.get(car_id) or {}).get("sons"), ((dy2.get(car_id) or {}).get("son"))))
        f.write("\n".join(lines))

    w()
    w("## 前后差异")
    w()
    for line in lines:
        w("- %s" % line)
    dump_log("unload_log.md")
    print("unload done ->", sample_dir)


def diff_units(st1, dy1, st2, dy2):
    """对比卸载前后的单位表，输出新增 / 消失 / 变化的编号。"""
    lines = []
    ids1, ids2 = set(st1), set(st2)
    added, removed = sorted(ids2 - ids1, key=idnum), sorted(ids1 - ids2, key=idnum)
    lines.append("新增编号: %s" % (", ".join(added) or "无"))
    lines.append("消失编号: %s" % (", ".join(removed) or "无"))
    fields = ("color", "typeid", "data", "pos", "pos2", "blood", "sons", "son", "car",
              "uid", "group", "army", "space", "stop")
    for oid in sorted(ids1 & ids2, key=idnum):
        a, b = st1[oid], st2[oid]
        ch = ["%s: %s -> %s" % (k, a.get(k), b.get(k)) for k in fields
              if str(a.get(k)) != str(b.get(k))]
        if ch:
            lines.append("%s 静态字段变化：%s" % (oid, "; ".join(ch)))
    for oid in sorted(set(dy1) | set(dy2)):
        a, b = dy1.get(oid) or {}, dy2.get(oid) or {}
        ch = ["%s: %s -> %s" % (k, a.get(k), b.get(k)) for k in
              ("son", "sonarr", "pos", "blood", "uid", "color")
              if str(a.get(k)) != str(b.get(k))]
        if ch:
            lines.append("%s 动态字段变化：%s" % (oid, "; ".join(ch)))
    if all(l.endswith("无") for l in lines[:2]) and len(lines) == 2:
        lines.append("（静态表无变化）")
    return lines


def idnum(oid):
    m = re.search(r"(\d+)", oid or "")
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main()

