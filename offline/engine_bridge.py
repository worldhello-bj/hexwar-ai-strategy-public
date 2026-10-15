# -*- coding: utf-8 -*-
"""把裁决引擎接入本地桩服务器。

每个 ASP 端点的返回值都按官方客户端的真实契约实现：
- com/updatePos.asp  → 返回该格机动耗时(毫秒)，客户端据此播放动画
- com/MoveEnd.asp    → 返回空串表示“继续机动”
- com/Attack.asp     → 返回非空表示“射击成功”，战果经下一帧 ObjAjax 下发
- RoomStart.asp      → 返回 Time1,WarDateTime,StartTime,.. （空串=失败）
- com/ComAjax.asp    → 返回 ComID|flag|{JSON}|时间|尾部，客户端据此执行指令
"""
import os
import re
import json
import time
import urllib.parse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Game, geometry as G
from engine import crt
from engine.equip import WEAPONS, EQUIPMENT, AMMO_FIELD

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(os.path.dirname(HERE), "snapshot")
RAW = os.path.join(SNAP, "raw")


def _read_raw(name):
    p = os.path.join(RAW, name)
    return open(p, "rb").read() if os.path.exists(p) else b""


def _read_user_fields():
    """取官方 `User_Get.asp` 的字段串（客户端据此拿 列/行 等）。

    ★ 本地完整采集（`snapshot/raw/`）**不入库**，公开克隆里没有；
      所以必须回退到随仓库发布的 `offline/mock/User_Get.asp`
      （同一份真实响应的桩，字段 6/7 就是 100×18）。
      否则 `base_fields` 只剩一个空串，客户端拿不到行列数、端到端校验会崩。
    """
    for p in (os.path.join(RAW, "User_Get.asp.txt"),
              os.path.join(RAW, "User_Get.asp"),
              os.path.join(HERE, "mock", "User_Get.asp")):
        if os.path.exists(p):
            return open(p, "rb").read().decode("gb18030", errors="replace").strip(), p
    return "", None


class Bridge:
    def __init__(self):
        self.g = Game()
        self.user_fields, self.user_fields_src = _read_user_fields()
        self.base_fields = self.user_fields.split("$")
        if len(self.base_fields) < 35:
            raise RuntimeError(
                "User_Get 字段串不可用（只有 %d 段）：请确认 offline/mock/User_Get.asp 存在"
                % len(self.base_fields))
        self.gamecolor = "NONE"
        self.pending = []          # 待下发指令
        self.com_log = []

    # ---------- 工具 ----------
    def _q(self, query, key, default=""):
        v = query.get(key, [default])
        return v[0] if isinstance(v, list) else v

    def _unit_fields(self):
        self.g.tick()
        f = list(self.base_fields)
        if len(f) > 34:
            f[20] = str(self.g.second)          # RoomSecond
            f[33] = self.g.now_str()            # ServerTime
            # Time1 / StartTime / WarDateTime 必须是日期串：
            # 客户端 GameTime() = SecDiff(ServerTime,StartTime) - SecDiff(Time1,WarDateTime)
            f[34] = self.g.start_str or ""      # Time1
            f[3] = self.g.start_str or ""       # StartTime
            f[11] = self.g.start_str or ""      # WarDateTime
        return f

    def log_line(self, s):
        self.com_log.append(s)

    # ---------- 主分发 ----------
    def handle(self, relpath, query, method, body):
        """返回 (content_bytes, ctype) 或 None（表示交给静态/桩文件）。"""
        p = relpath.lower()

        if p == "user_get.asp":
            return ("$".join(self._unit_fields())).encode("gb18030"), "text/html; charset=gbk"

        if p == "roomstart.asp":
            if not self.g.started:
                self.g.start()
            self.g.record("推演开始")
            t = self.g.start_str
            payload = "%s,%s,%s,%s,%s" % (t, t, t, "", "")
            return payload.encode("gb18030"), "text/html; charset=gbk"

        if p == "roomstat.asp":
            if not self.g.started:
                return b"", "text/html; charset=gbk"
            t = self.g.start_str
            payload = "%s,%s,%s" % (t, t, t)
            return payload.encode("gb18030"), "text/html; charset=gbk"

        if p == "getservertime.asp":
            return self.g.now_str().encode("gb18030"), "text/html; charset=gbk"

        if p == "objajax.asp":
            self.g.tick()
            obj = {}
            for u in self.g.units.values():
                obj[u.id] = self.g.obj_state(u)
            s = json.dumps(obj, ensure_ascii=False)
            return s.encode("gb18030"), "text/html; charset=gbk"

        if p == "report_attack.asp":
            txt = self.g.log[-1] if self.g.log else "暂无战报"
            return (txt + "|").encode("gb18030"), "text/html; charset=gbk"

        if p == "objfire.asp":
            oid = self._q(query, "ObjID")
            u = self.g.units.get(oid)
            if u:
                u.att = 0
            # 第4段 <=0 时客户端会走 loadFireData(c) 而不是 ObjFire(...)，
            # 避免因找不到临时弹道 DOM 而抛异常。
            return "0,0,0,0".encode("gb18030"), "text/html; charset=gbk"

        # ---------- 指令通道 ----------
        if p == "com/compost.asp":
            return self._com_post(query)

        if p == "com/comajax.asp":
            return self._com_ajax()

        if p == "com/updatepos.asp":
            return self._update_pos(query)

        if p == "com/moveend.asp":
            return b"", "text/html; charset=gbk"

        if p == "com/attack.asp":
            return self._attack(query)

        if p == "com/citytake.asp":
            return self._city_take(query)

        if p in ("com/comflag.asp", "com/comflagtime.asp", "com/comtime.asp",
                 "com/comdelete.asp", "com/objstop.asp", "com/objmovestop.asp",
                 "com/objattstop.asp", "com/objweaponready.asp", "com/objkeepcancel.asp",
                 "com/objkeepremove.asp", "com/objpassremove.asp", "com/objradar.asp",
                 "com/commerge.asp", "com/objinter.asp", "com/objbl.asp",
                 "com/objwork.asp", "com/objworkready.asp", "com/objsz.asp",
                 "com/objtalkready.asp", "com/objmovearr.asp", "com/objtaskdel.asp",
                 "com/commandctrl.asp", "com/roomtaskpost.asp", "com/objstepToggle.asp",
                 "com/objheight.asp", "com/fortobj.asp", "com/jmremove.asp",
                 "com/attack.asp", "com/obj2_out.asp", "com/obj2_out2.asp",
                 "com/objinto.asp", "com/objmerge.asp", "com/objbl.asp",
                 "com/moveobject.asp", "com/comdelete.asp", "com/objc0.asp",
                 "com/comattack.asp", "com/kill.asp", "com/lcattack.asp",
                 "com/roaddes.asp", "com/gasremove.asp", "com/objcomm.asp",
                 "com/objf1.asp", "com/objf2.asp", "com/objweapon.asp"):
            return b"", "text/html; charset=gbk"

        if p == "com/tglist.asp":
            return self._tg_list(query)

        if p == "objinfo.asp":
            return self._obj_info(query)

        if p == "obj_weapon.asp" or p == "com/commandobj.asp":
            return self._obj_weapon(query, p)

        if p == "com/objtask.asp":
            return self._obj_task(query)

        if p == "com/commandsow.asp" or p == "com/commandshow.asp":
            return b"", "text/html; charset=gbk"

        return None

    # ---------- 指令提交 ----------
    def _com_post(self, query):
        oid = self._q(query, "ObjID")
        raw = self._q(query, "Command")
        task = self._q(query, "TaskName")
        u = self.g.units.get(oid)
        if not u:
            return b"", "text/html; charset=gbk"

        parsed = None
        try:
            parsed = json.loads(raw) if raw else None
        except Exception:
            parsed = None

        cmds = {}
        if isinstance(parsed, dict):
            cmds = parsed
        elif raw and re.match(r"^\s*\{", raw):
            try:
                cmds = json.loads(re.sub(r"(\d+):", r'"\1":', raw))
            except Exception:
                cmds = {}
        if not cmds:
            # 交互式机动：客户端会在 ComAjax 里收到 move 指令
            self.log_line("收到空指令: %s" % task)
            return b"", "text/html; charset=gbk"

        for target_id, c in cmds.items():
            tid = target_id if target_id in self.g.units else \
                next((k for k, v in self.g.units.items() if v.id == target_id), target_id)
            tu = self.g.units.get(tid)
            if not tu or not isinstance(c, dict):
                continue
            cid = self.g.new_command(c)
            if c.get("move"):
                self.pending.append((cid, tid, c))
                self.log_line("机动指令 #%d → %s 路径 %s" % (cid, tu.army or tu.name, c.get("move")))
            elif c.get("attack"):
                tgt = c["attack"]
                if tgt not in self.g.units:
                    tgt = next((k for k, v in self.g.units.items() if v.id == tgt), tgt)
                r, err = self.g.attack(tid, tgt, weapon_id=int(c.get("weapon") or 0))
                self.log_line("射击指令 #%d → %s" % (cid, err or "已裁决"))
            elif c.get("flag"):
                cid2 = cid
                if self.g.apply_city_capture(tu):
                    self.log_line("夺控指令执行")
            elif c.get("stat") is not None:
                tu.ste = 2 if int(c["stat"]) == 2 else 0
        return b"", "text/html; charset=gbk"

    def _com_ajax(self):
        """把待执行指令下发给客户端。"""
        if not self.pending:
            return b"", "text/html; charset=gbk"
        cid, oid, c = self.pending.pop(0)
        payload = {}
        payload[oid] = {
            "move": c.get("move", ""),
            "speed": c.get("speed", 0),
        }
        body = "%d|0|%s||0" % (cid, json.dumps(payload, ensure_ascii=False))
        return body.encode("gb18030"), "text/html; charset=gbk"

    def _update_pos(self, query):
        oid = self._q(query, "ObjID")
        mid = self._q(query, "MapID")
        u = self.g.units.get(oid)
        if not u or not mid.isdigit():
            return b"0", "text/html; charset=gbk"
        to_id = int(mid)
        sec, why = self.g.commit_step(u, u.pos, to_id)
        if sec <= 0:
            return b"0", "text/html; charset=gbk"
        u.pos = to_id
        self.g.dirty = True
        ms = int(sec * 1000)
        return str(ms).encode("gb18030"), "text/html; charset=gbk"

    def _attack(self, query):
        oid = self._q(query, "ObjID")
        tid = self._q(query, "TarID")
        wid = self._q(query, "WeaponID", "0")
        cid = self._q(query, "ComID", "0")
        try:
            wid = int(wid)
        except Exception:
            wid = 0
        r, err = self.g.attack(oid, tid, weapon_id=wid)
        if err:
            self.log_line("射击失败: %s" % err)
            return b"", "text/html; charset=gbk"
        return b"1", "text/html; charset=gbk"

    def _city_take(self, query):
        oid = self._q(query, "ObjID")
        u = self.g.units.get(oid)
        if u and self.g.apply_city_capture(u):
            return b"1", "text/html; charset=gbk"
        return b"", "text/html; charset=gbk"

    # ---------- 面板 ----------
    def _tg_list(self, query):
        """返回可打击目标列表（JSON）。官方此接口 404，这里给出可用实现。"""
        oid = self._q(query, "ObjID")
        u = self.g.units.get(oid)
        if not u:
            return b"", "text/html; charset=gbk"
        out = []
        for t in self.g.units.values():
            if not t.alive() or t.color == u.color:
                continue
            d = G.distance(u.pos, t.pos)
            if d > 20:
                continue
            ok, _ = self.g.los(u.pos, t.pos)
            if not ok or not self.g.can_observe(oid, t.id):
                continue
            out.append({"id": t.id, "name": t.army or t.name, "dist": d})
        if not out:
            return b"", "text/html; charset=gbk"
        return json.dumps(out, ensure_ascii=False).encode("gb18030"), "text/html; charset=gbk"

    def _obj_info(self, query):
        oid = self._q(query, "ObjID")
        u = self.g.units.get(oid)
        if not u:
            return b"", "text/html; charset=gbk"
        ws = []
        for wid in u.wz:
            w = WEAPONS.get(wid)
            if w:
                fld = AMMO_FIELD.get(w[1])
                n = u.ammo.get(fld, 0) if fld else 0
                ws.append("%s %s %d格 %d" % (w[0], w[1], w[2], n))
        eq = EQUIPMENT.get(u.data, ("", 0, "", 0, (0, 0, 0), []))
        html = ("战斗编组 %s 编制序列 %s 棋子类型 %s 装甲防护 %s 权重分值 %d "
                "侦察能力 人员目标 %d格 车辆目标 %d格 空中目标 %d格 "
                "武器装备 %s" % (u.group or "-", u.army or "-", u.name, u.armor, u.score,
                              eq[4][0], eq[4][1], eq[4][2], " / ".join(ws)))
        return html.encode("gb18030"), "text/html; charset=gbk"

    def _obj_weapon(self, query, path):
        oid = self._q(query, "ObjID")
        u = self.g.units.get(oid)
        if not u or not u.wz:
            return b"", "text/html; charset=gbk"
        items = []
        for wid in u.wz:
            w = WEAPONS.get(wid)
            if not w:
                continue
            fld = AMMO_FIELD.get(w[1])
            n = u.ammo.get(fld, 0) if fld else 0
            items.append('<label class="weapon">%s [弹药:%d 射程:%d]</label>' % (w[0], n, w[2]))
        html = "武器 " + " ".join(items)
        return html.encode("gb18030"), "text/html; charset=gbk"

    def _obj_task(self, query):
        oid = self._q(query, "ObjID")
        u = self.g.units.get(oid)
        if not u:
            return b"", "text/html; charset=gbk"
        rows = []
        for cid, o in self.g.orders.items():
            d = o.get("data") or {}
            rows.append("<div>#%d %s</div>" % (cid, json.dumps(d, ensure_ascii=False)))
        return ("".join(rows)).encode("gb18030"), "text/html; charset=gbk"

    # ---------- 计分 ----------
    def score(self):
        red = sum(u.score for u in self.g.units.values() if u.color == "RED" and u.alive())
        blue = sum(u.score for u in self.g.units.values() if u.color == "BLUE" and u.alive())
        cap_red = sum(1 for c in self.g.cities if c.get("flag") == "RED")
        cap_blue = sum(1 for c in self.g.cities if c.get("flag") == "BLUE")
        return red, blue, cap_red, cap_blue
