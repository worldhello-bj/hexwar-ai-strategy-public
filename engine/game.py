# -*- coding: utf-8 -*-
"""离线裁决引擎：状态、规则与指令处理。

设计要点（来自对官方客户端的实测与手册第 4 章）：
- 引擎是唯一权威；客户端每秒通过 ObjAjax.asp 覆盖式同步。
- 机动指令由客户端本地播放动画，服务端在 com/updatePos.asp 返回本格耗时(ms)。
- 直瞄射击由 com/Attack.asp 裁决，结果通过下一帧 ObjAjax 下发（血量/压制变化）。
"""
import os
import json
import math
import time
import random

from . import geometry as G
from . import crt
from . import paths as _paths
from .equip import (EQUIPMENT, WEAPONS, AMMO_FIELD, ANTI_AIR_CLASS, CRT_AIR_NAME,
                    carried_establishment)

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = (os.environ.get("HEXWAR_SNAP")
        or _paths.snapshot_dir())

# 压制持续（手册 4.6.3：150 秒）
SUPPRESS_SEC = 150
# 疲劳移除时长（想定参数，取 SetUp.js 默认口径）
FATIGUE_SEC = 60
# 夺控判定所需兵力
CAPTURE_MIN_BLOOD = 1

# ---------------- 巡飞弹生命周期常量（手册 4.10 / issue #1） ----------------
# 巡飞超时上限：手册 4.10(5) 只说"超时自毁"，**未给数值**。平台取值有代码级证据
# （`offline/assets/war.min.js`）：
#     FlyTime = parseInt(1200 / RoomSpeed)
#     巡飞计时 = parseInt((new Date(ServerTime) - new Date(radar)) / 1000)   # 秒
#     计时 >= FlyTime → CommandKill(弹)，并把弹的 `car`（载车）的 `stop` 清 0
# ⇒ "巡飞上限 = 1200 / RoomSpeed 秒"，且这里的"秒"与 `ServerTime` 差值同尺度
#   （客户端每秒轮询一次、`RoomSecond` 每秒 +1，与本引擎 `Game.second` 同尺度）。
# ⚠️ **仍未确证**：真实发射后服务端下发的 `radar`/`car` 取值，以及 RoomSpeed 与
#   `RoomSecond` 的换算。已知的先例是「房间时长 × RoomSpeed」
#   （`KeepSec 50 × RoomSpeed 3 = SUPPRESS_SEC 150`），与 FlyTime 的**除法方向相反**
#   ⇒ 两者是否同一口径没有实测样本。所以这里**不写死 1200**：默认按客户端公式算
#   （RoomSpeed=3 → 400 秒），允许显式覆盖，并把每个在飞弹的取值与来源记进 `detail`。
LOITER_ROOM_UNITS = 1200        # 客户端 FlyTime 的分子
DEFAULT_ROOM_SPEED = 3          # 练习服/比赛服实测 RoomSpeed（`User_Get[16]`）
LOITER_SEC_PLACEHOLDER = 120    # 早前的无来源占位值，保留仅作对照（不再是默认）

# 巡飞弹的显式生命周期状态（见 `Game.missile_state`）。
MISSILE_STATES = ("unassigned", "carried", "loitering", "spent", "lost")

# 部署（上车）窗口（手册 4.10(3)：“需在对抗开始前上车确定所属关系”）：
#   "opening"（默认）—— 只接受 `second == 0`，即本环境里唯一能代表“对抗开始前”的时刻；
#   "any"            —— 只要该弹**尚未发射**（仍为 `unassigned`）就允许部署。
# 环境没有赛前阶段（`reset()` 后 `started` 已为真），所以“开局窗口”只是一个**口径**：
# 默认取忠实手册的 `opening`，要训练里能学到用弹就显式传 `any`（不静默偏离）。
DEPLOY_WINDOWS = ("opening", "any")


class Unit:
    __slots__ = ("id", "data", "name", "army", "group", "typeid", "color", "pos",
                 "pos2", "rot", "blood", "space", "score", "armor", "units",
                 "wz", "ammo", "stunned_until", "fatigued_until", "missile_ids",
                 "uid", "ste", "keep", "pass_", "att", "ai", "c0", "c3", "detail",
                 "radar", "see", "task", "alt", "establishment")

    def __init__(self, d):
        self.id = d["id"]
        self.data = int(d.get("data") or 0)
        eq = EQUIPMENT.get(self.data, ("未知", 1, "无装甲", 0, (0, 0, 0), []))
        self.name = d.get("name") or eq[0]
        self.army = d.get("army") or ""
        self.group = d.get("group") or ""
        self.typeid = int(d.get("typeid") or eq[1])
        self.color = d.get("color") or ""
        self.pos = int(d.get("pos") or 0)
        self.pos2 = int(d.get("pos2") or 0)
        self.rot = int(d.get("rot") or 0)
        self.blood = int(d.get("blood") or 0)
        # 编制兵力（战前的应有兵力/车数）。用于计分的"损失"计算：
        # 上阵单位 = 载入时记录的 blood；载员（在车内时 blood=0）在 `engine.scoring`
        # 里按装备表补上（见 docs/计分规则待确认.md）。
        self.establishment = self.blood
        self.space = int(d.get("space") or 0)
        self.score = int(eq[3] if d.get("score") in (None, "") else d["score"])
        self.armor = eq[2]
        self.units = max(1, int(d.get("blood") or 1)) if self.typeid in (1, 2) else 1
        self.wz = [int(x) for x in (d.get("wz") or "").replace(" ", "").split(",") if x.strip().isdigit()]
        if not self.wz:
            self.wz = list(eq[5])
        self.ammo = {"a3": int(d.get("a3") or 0), "a4": int(d.get("a4") or 0),
                     "a5": int(d.get("a5") or 0)}
        self.stunned_until = 0.0
        self.fatigued_until = 0.0
        self.missile_ids = []
        self.uid = int(d.get("uid") or 0)
        self.ste = int(d.get("ste") or 0)
        self.keep = int(d.get("keep") or 0)
        self.pass_ = int(d.get("pass") or 0)
        self.att = int(d.get("att") or 0)
        self.ai = int(d.get("ai") or 0)
        self.c0 = int(d.get("c0") or 0)
        self.c3 = int(d.get("c3") or 0)
        self.radar = d.get("radar") or ""
        self.see = int(d.get("see") or 0)
        self.task = int(d.get("task") or 0)
        # 空中单位滞空高度码（手册 4.8(2)：500 高空 / 200 低空 / 20 超低空）
        self.alt = int(d.get("e2") or 0)
        self.detail = {}
        # 载员-载具配对证据（仅用于 `Game.link_carried`，不参与裁决）：
        # 载员的 `ico` 与载具 `son` 图标串里的同名字段对应（实测：COD-M4 ↔ M113TW、
        # PLA2-191 / UGVS / YY4 / WS43 ↔ ZBD100）。
        if d.get("ico"):
            self.detail["ico"] = str(d["ico"])
        icons = [x for x in str(d.get("son") or "").split(",") if x]
        if icons:
            self.detail["son_icons"] = icons
        # 手册 4.10 巡飞弹 / 装卸的**原始生命周期字段**：`car`（所属载具）与 `stop`
        # （载车的"在飞占用"标志）。原实现直接丢弃 ⇒ 生命周期只能靠猜（issue #1 第 5 条）。
        # 这里只**保留证据**，不做任何静默推断，且"无值"不写进 detail：
        # 平台里 `car=Obj_0` 就是"没有载车"、`stop=0` 就是"没有在飞占用"；
        # 想定 7527 的 36 个单位全都是 `car=Obj_0` ⇒ 不能凭"同格"或"武器 76"推断已部署。
        car = str(d.get("car") or "")
        if car and car != "Obj_0":
            self.detail["car"] = car
        stop = int(d.get("stop") or 0)
        if stop:
            self.detail["stop"] = stop

    # --- 单位类别（手册 4.8 / 4.9 / 4.10）---
    def is_air(self):
        return self.typeid == 3

    def is_loitering_missile(self):
        """巡飞弹（手册 4.10）：typeid=3 中的「巡飞弹」。"""
        return self.name == "巡飞弹"

    def is_ugv(self):
        """无人战车（手册 4.9）：地面无人平台。按名称前缀「无人」识别。"""
        return self.typeid == 2 and (self.name or "").startswith("无人")

    def can_disembark(self):
        """手册 3.3 的**下车**对象：人员与无人战车（`typeid` 1 / 2）。

        巡飞弹与其它空中弹（`typeid` 3）**不能下车**：客户端「全员下车」只连发
        `off=1`（人员）/ `off=2`（无人车），而 `off=3` 走的是 `com/Obj2_Out.asp` +
        `CommandOff` —— 那是**发射**命令（要求载车 `stop != 1`，且开始前不允许）。
        所以弹离开载车只有**发射**一条路；中途把它“取回”（回到未部署）在平台上并不存在。
        """
        return self.typeid in (1, 2)

    def out_of_control(self):
        """手册 4.9(3)：无人战车失去有人战车依托 → 失控。"""
        return bool(self.detail.get("out_of_control"))

    # --- 便于裁决的派生属性 ---
    def moving(self, now):
        return self.detail.get("move_until", 0) > now

    def stunned(self, now):
        return self.stunned_until > now

    def fatigued(self, now):
        return self.fatigued_until > now

    def alive(self):
        return self.blood > 0

    def weapon_name(self, idx=0):
        if not self.wz:
            return None
        wid = self.c3 if self.c3 in self.wz else self.wz[min(idx, len(self.wz) - 1)]
        w = WEAPONS.get(wid)
        return w[0] if w else None

    def is_anti_air_capable(self):
        for wid in self.wz:
            w = WEAPONS.get(wid)
            if w and w[0] in ANTI_AIR_CLASS:
                return True
        return False


class Game:
    """一局推演。完全本地，无外部依赖。"""

    def __init__(self, seed=None, snapshot=None, room_speed=None, loiter_sec=None,
                 deploy_window=None):
        if room_speed is not None and (isinstance(room_speed, bool)
                                       or not isinstance(room_speed, (int, float))
                                       or room_speed <= 0):
            raise ValueError("room_speed must be a positive number")
        if loiter_sec is not None and (isinstance(loiter_sec, bool)
                                       or not isinstance(loiter_sec, (int, float))
                                       or loiter_sec <= 0):
            raise ValueError("loiter_sec must be a positive number")
        if deploy_window is None:
            deploy_window = "opening"
        if deploy_window not in DEPLOY_WINDOWS:
            raise ValueError("deploy_window must be one of %s" % (DEPLOY_WINDOWS,))
        self.rng = random.Random(seed)
        # 巡飞弹部署窗口（见 `DEPLOY_WINDOWS`）：默认“对抗开始前”。
        self.deploy_window = deploy_window
        # 巡飞超时上限（秒）：默认按客户端公式 `parseInt(1200 / RoomSpeed)` 算，
        # 允许显式覆盖；取值与来源逐弹记进 `detail`（**不假装已实测**）。
        self.room_speed = DEFAULT_ROOM_SPEED if room_speed is None else room_speed
        if loiter_sec is None:
            self.loiter_sec = self.loiter_seconds(self.room_speed)
            self.loiter_sec_source = "client:parseInt(1200/RoomSpeed)"
        else:
            self.loiter_sec = loiter_sec
            self.loiter_sec_source = "explicit override"
        snap = snapshot or SNAP
        self.snap_dir = snap
        self.hexes = {}
        p = os.path.join(snap, "map_hexes.json")
        with open(p, encoding="utf-8") as stream:
            raw = json.load(stream)
        for k, v in raw.items():
            self.hexes[int(k)] = v
        with open(os.path.join(snap, "cities.json"), encoding="utf-8") as stream:
            self.cities = json.load(stream)
        for c in self.cities:
            c["pos"] = int(c.get("pos") or 0)
        self.units = {}
        with open(os.path.join(snap, "units.json"), encoding="utf-8") as stream:
            raw_units = json.load(stream)
        for u in raw_units:
            uu = Unit(u)
            self.units[uu.id] = uu
        # ★ 不在这里缓存地形派生量：`hexes` 是公开可变字典（测试/核查脚本会原地改
        #   `step`/`elevation` 验证规则），按 mapid 缓存会拿到旧值。只保留了
        #   “cond → 地形名” 这个单标量纯函数的去掉分支优化（见 `_TERRAIN_BY_COND`）。
        # 快照里的「人在车内」记录只有 `pos2`（载具所在格），必须在载入时把关系
        # 固定到**载具 ID** 上；否则载具一机动，`pos2` 变旧格，载员的分值就会
        # 从计分里凭空消失（等于惩罚机动）。见 `docs/计分规则待确认.md`。
        self.link_carried()
        # 数据里**显式写明**的归属（`car` = 载具 ID）也要在载入时落成车内状态：
        # 这是“初设即已部署上车”的表达方式（与上面那一步的**证据推断**不同）。
        self.link_declared_carriers()

        self.started = False
        self.t0 = 0.0          # 推演开始时刻
        self.second = 0        # 已推演秒数
        self.round = 0
        self.log = []
        self.orders = {}       # ComID -> 指令
        self.com_seq = 0
        self.fires = {}        # ObjID -> 射击特效
        self.dirty = True
        self.obj_fire_seq = {}
        self.start_str = ""    # 开始时刻的日期串（客户端时钟用）

    # ---------------- 载员-载具关联 ----------------
    def link_declared_carriers(self):
        """把数据里**显式声明**的所属关系（`car` 字段）落成车内状态。

        平台想定里可以用 `car=<载具ID>` + `pos=0` / `pos2=<载具格>` / `blood=0` / `color=''`
        表达“**开局就在车内**”（初设即已部署上车）。本函数只读这个**写明的**字段，
        不做任何推断（同格、载具 `son` 图标串、武器 76 都**不**构成归属证据 ——
        issue #1 明确禁止静默推断；那段逻辑只在 `link_carried()` 里处理“无阵营、
        `pos=0`、`blood=0`”的旧快照记录，且要求唯一命中）。

        `car=Obj_0`（本仓库初设的取值）或空串 ⇒ 无归属，**不会**建立任何关系。
        返回本次落成的单位 id 列表（供核对/测试；`units.json` 未声明时为空）。
        """
        self.carried_declared = []
        self.carried_declare_unlinked = []
        for u in sorted(self.units.values(), key=lambda x: str(x.id)):
            cid = u.detail.get("car")
            if not cid or u.detail.get("carrier") or cid == u.id:
                continue
            host = self.units.get(cid)
            if host is None:
                # 写了归属但载具不在场：不静默忽略，记下来由调用方/测试报出。
                self.carried_declare_unlinked.append((u.id, cid))
                continue
            u.detail["carrier"] = host.id
            host.detail.setdefault("carried", []).append(u.id)
            # 车内读不到兵力与阵营（平台口径）：编制留在 `establishment`。
            u.establishment = u.establishment or u.blood
            u.blood = 0
            u.color = ""
            if u.is_loitering_missile():
                u.detail["missile_state"] = "carried"
            if u.pos:
                # 车内记录的平台形态：`pos=0`、`pos2`= 载具格。
                u.pos2 = u.pos2 or host.pos
                u.pos = 0
            self.carried_declared.append(u.id)
        return list(self.carried_declared)

    def link_carried(self):
        """把快照里「人在车内」的记录与载具建立**稳定**关联。

        写 `carried.detail["carrier"] = 载具ID` 与 `carrier.detail["carried"]`，
        与装卸动作（`env.hexwar_env` 的 load/unload）用的是同一套字段。

        为什么必须在载入时固化：快照只给 `pos2`（当时的载具格），而 `pos2` 是
        「目标格/旧格」，载具一机动就不再指向它 —— 若靠 `pos2` 现场匹配，载具
        离开原格后载员就会从计分里**凭空消失**，等价于对机动施加惩罚。
        另外 `pos2` 也可能同时落在多辆载具上（堆叠），所以要按证据强弱定序：

        ① 载具 `son` 图标串含该载员的 `ico`（图标一位一用，避免同型载员重复占位）；
        ② 该格唯一的车辆/飞行器（`typeid` 2/3）；
        ③ 该格唯一的单位。

        ①②③ 都定不下去时按 `id` 排序取第一辆，并**显式记入 `self.carried_ambiguous`**；
        完全找不到归属的记入 `self.carried_unlinked` —— 两者都由 `engine.scoring`
        报出（不静默算分）。重复调用是安全的（已关联的不再处理）。
        """
        self.carried_linked = []
        self.carried_unlinked = []
        self.carried_ambiguous = []
        pending = [u for u in self.units.values()
                   if not u.color and u.pos == 0 and u.blood == 0 and u.pos2
                   and not u.detail.get("carrier")]
        hosts = [u for u in self.units.values() if u.color]
        for u in sorted(pending, key=lambda x: str(x.id)):
            group = [h for h in hosts if h.pos == u.pos2]
            icons_left = {h.id: list(h.detail.get("son_icons") or []) for h in group}
            ico = u.detail.get("ico")
            candidates = [h for h in group if ico and ico in icons_left[h.id]]
            if len(candidates) > 1:
                self.carried_ambiguous.append((u.id, [h.id for h in candidates]))
            if not candidates:
                vehicles = [h for h in group if h.typeid in (2, 3)]
                candidates = vehicles or group
                if len(candidates) > 1:
                    self.carried_ambiguous.append((u.id, [h.id for h in candidates]))
            if not candidates:
                self.carried_unlinked.append(u.id)
                continue
            host = sorted(candidates, key=lambda h: str(h.id))[0]
            if ico in icons_left[host.id]:
                icons_left[host.id].remove(ico)
            u.detail["carrier"] = host.id
            host.detail.setdefault("carried", []).append(u.id)
            self.carried_linked.append(u.id)
        return {"linked": list(self.carried_linked),
                "unlinked": list(self.carried_unlinked),
                "ambiguous": list(self.carried_ambiguous)}

    # ---------------- 时钟 ----------------
    def start(self):
        self.started = True
        self.t0 = time.time() - self.second
        self.start_str = self.now_str()
        return True

    def tick_second(self):
        """推进 **1 秒**，并按**秒**粒度结算所有自动规则。

        这是与官方一致的粒度：服务端每秒刷新一次（`ObjAjax` / `User_Get`），
        玩家每一秒都能观察 / 下令 / 改令。`advance()` 与 `tick()` 都必须**逐秒**走这一函数——
        否则自动裁决次数会取决于"调用了几次"而不是"过了几秒"。

        实测（同一段 60 秒、同一个持续打击目标）：
        逐秒推进打出 **41** 发直瞷裁决，一次跳 60 秒只打出 **3** 发。
        """
        self.second += 1
        self.t0 = time.time() - self.second
        self.resolve_engagements()  # 手册 3.14 持续打击（内部按 ATTACK_INTERVAL_SEC 门控）
        self.same_hex_engage()      # 手册 4.11 同格交战（内部按 ENGAGE_INTERVAL_SEC 门控）
        self._expire_states()
        return self.second

    def tick(self):
        """实时时钟：把推演推进到真实经过的秒数（**逐秒**结算，不是一次算完）。"""
        if not self.started:
            return self.second
        target = int(time.time() - self.t0)
        while self.second < target:
            self.tick_second()
        return self.second

    def advance(self, seconds):
        """把推演时钟向前推进若干秒（供无头模拟/策略搜索使用，与真实时钟无关）。

        ★ **逐秒推进**：`advance(N)` 必须与 `advance(1)×N` 完全等价，
        否则“把动作拆碎”就能多换自动裁决（实测 41 发 vs 3 发）——那是可以刷分的偏差。
        """
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("seconds must be a finite number")
        if seconds <= 0:
            return self.second
        for _ in range(int(math.ceil(seconds))):
            self.tick_second()
        return self.second

    def _expire_states(self):
        now = self.second
        for u in self.units.values():
            if u.stunned_until and u.stunned_until <= now:
                u.stunned_until = 0.0
                u.keep = 0
                # 手册 4.1.1(6)：解压制后继续执行未完成的机动命令
                if u.detail.get("path"):
                    self.record("%s 压制解除，继续执行未完成的机动（手册 4.1.1(6)）"
                                % (u.army or u.name))
            if u.fatigued_until and u.fatigued_until <= now:
                u.fatigued_until = 0.0
            if u.detail.get("path") and not u.stunned(now):
                self._resume_path(u, now)
        self._update_dependencies()
        self._expire_missiles(now)

    # -------- 手册 4.1.1(6) / 4.9 / 4.10 --------
    def begin_move(self, unit_id, path):
        """登记一条机动路径（相邻格序列）。

        手册 4.1.1(6)：机动中遭压制 → 完成当前格后停止，压制解除后**继续执行**
        未完成的机动。登记后由 `_resume_path` 在每次 tick 推进一格。
        """
        u = self.units.get(unit_id)
        if not u:
            return False, "棋子不存在"
        u.detail["path"] = [int(x) for x in path]
        u.detail["path_idx"] = 0
        u.detail["move_until"] = 0          # 新指令：清掉上一条路径遗留的到达时刻
        return True, ""

    def pending_moves(self):
        """仍有未完成机动路径的单位 id 列表。"""
        return [u.id for u in self.units.values() if u.detail.get("path")]

    def _resume_path(self, u, now):
        path = u.detail.get("path") or []
        i = u.detail.get("path_idx", 0)
        if i >= len(path):
            u.detail.pop("path", None)
            u.detail.pop("path_idx", None)
            u.detail.pop("move_until", None)
            return
        # ★ 按**秒**推进：本格还没走完就不动。缺这道门，推进速度会变成
        #   “每个调用一格”（实测 60 秒里 60 次调用走 8 格，而规则是 60 秒/格）。
        if now < u.detail.get("move_until", 0):
            return
        to_id = path[i]
        sec, why = self.step_seconds(u, u.pos, to_id)
        if sec <= 0:
            if "被压制" in why:
                return                       # 仍在压制中，等下一 tick
            self.record("%s 无法继续机动：%s" % (u.army or u.name, why))
            u.detail.pop("path", None)
            u.detail.pop("path_idx", None)
            u.detail.pop("move_until", None)
            return
        u.pos = to_id
        u.detail["move_until"] = now + sec
        u.detail["path_idx"] = i + 1
        if i + 1 >= len(path):
            u.detail.pop("path", None)
            u.detail.pop("path_idx", None)
        self.record("%s 机动至 %d（%.0f 秒）" % (u.army or u.name, to_id, sec))

    def _update_dependencies(self):
        """手册 4.9(3)：无人战车所属有人战车被摧毁 → 失控（不能射击/侦察/校射）。"""
        for u in self.units.values():
            if not u.is_ugv() or u.detail.get("out_of_control"):
                continue
            cid = u.detail.get("carrier")
            if not cid:
                continue
            c = self.units.get(cid)
            if c is None or not c.alive():
                u.detail["out_of_control"] = True
                self.record("%s 依托的有人战车 %s 被摧毁，失去控制（手册 4.9(3)）"
                            % (u.army or u.name, cid))
        # 手册 4.10(5)：载车被摧毁时，仍**在车内**（尚未发射）的巡飞弹随之损失。
        # 显式置 `lost`：否则它会保持 `carried` 状态，看上去“还能发射”（不复活）。
        for u in self.units.values():
            if not u.is_loitering_missile() or self.missile_state(u) != "carried":
                continue
            c = self.units.get(u.detail.get("carrier"))
            if c is None or not c.alive():
                u.detail["missile_state"] = "lost"
                self.record("载车被摧毁，车内巡飞弹 %s 损失（手册 4.10(5)）" % u.id)

    # ---------------- 手册 4.10 巡飞弹生命周期（issue #1） ----------------
    def missile_state(self, m):
        """巡飞弹的显式生命周期状态（`engine.game.MISSILE_STATES` 之一）。

        | 状态 | 含义 | 依据 |
        |---|---|---|
        | `unassigned` | 在图上、与载车**没有**所属关系（想定 7527 初态全是这种：36 个单位都是 `car=Obj_0`） | 手册 4.10(3) |
        | `carried` | 已在载车内（`detail["carrier"]`；车内读不到兵力 ⇒ `blood=0`、阵营为空） | 平台装卸实测 |
        | `loitering` | 已发射、在飞（`detail["launcher"]`） | 手册 4.10(4) |
        | `spent` | 已完成一次攻击并消耗（负责人 2026-10-16 确认：攻击后巡飞弹消失） | 负责人确认 |
        | `lost` | 已损失（巡飞超时自毁 / 载车被毁 / 外部击杀） | 手册 4.10(5) |

        状态优先取 `detail["missile_state"]`（显式记录），缺记录时推导。
        ★ 判据顺序很重要：**先看载车、再看血量** —— 车内记录 `blood=0` 是“车内读不到兵力”，
        不是“已损失”（装卸上已经踩过一次这个坑）。所以 `carrier` + `blood=0` = `carried`。
        """
        state = m.detail.get("missile_state")
        if state in ("spent", "lost"):
            return state
        if m.blood <= 0 and not m.detail.get("carrier"):
            # 血量归零 = 已经不在图上（车内记录 blood=0 是例外：那是“车内读不到兵力”）。
            return "lost"
        if state in MISSILE_STATES:
            return state
        if m.detail.get("launcher"):
            return "loitering"
        if m.detail.get("carrier"):
            return "carried"
        return "unassigned"

    def in_flight_missile(self, carrier_id):
        """该载车当前**在飞**的巡飞弹（没有则 None）。**纯查询，不修改状态。**"""
        c = self.units.get(carrier_id)
        if c is None:
            return None
        m = self.units.get(c.detail.get("launched_missile"))
        if m is None or self.missile_state(m) != "loitering":
            return None
        return m

    def release_launcher(self, carrier_id):
        """解除载车的“在飞占用”（`stop` 归零），使该车可再次发射。

        手册 4.10(4) 的约束是“**同时**只能有一枚在飞”，不是“一局一枚”
        （issue #1 第 3 条：原实现的 `launched_missile` 是永久锁）。
        """
        c = self.units.get(carrier_id)
        if c is None:
            return
        c.detail.pop("launched_missile", None)
        if c.detail.get("stop"):
            c.detail["stop"] = 0

    def missile_deploy_block(self, carrier_id, missile_id):
        """部署巡飞弹的**纯**规则前置条件（不修改任何状态）。

        返回 `None` 表示允许，否则返回拒绝原因。`deploy_missile()` 与环境的**只读**枚举
        （`HexwarEnv._load_gate`）共用同一条口径，避免两份判定漂移。
        """
        c = self.units.get(carrier_id)
        m = self.units.get(missile_id)
        if not c or not m:
            return "单位不存在"
        if not m.is_loitering_missile():
            return "该棋子不是巡飞弹"
        state = self.missile_state(m)
        if state != "unassigned":
            return "巡飞弹当前状态为 %s，不能部署上车" % state
        if not self.started:
            return "对抗尚未开始"
        if self.deploy_window == "opening" and self.second != 0:
            return "巡飞弹须在对抗开始前上车（手册 4.10(3)）"
        if not c.alive():
            return "载车已被消灭"
        if c.typeid != 2:
            return "巡飞弹只能部署在车辆上"
        if not c.color or c.color != m.color:
            return "巡飞弹只能部署在己方车辆上"
        if c.pos != m.pos:
            return "巡飞弹与载车必须同格（手册 3.3）"
        if c.stunned(self.second) or m.stunned(self.second):
            return "被压制不能上/下车（手册 4.6.3(1)）"
        return None

    def deploy_missile(self, carrier_id, missile_id):
        """手册 4.10(3)：巡飞弹须在**对抗开始前**上车以确定所属关系。

        显式转换 `unassigned → carried`，**不做任何静默推断**（同格、`son` 图标、
        武器 76 都不构成“已部署/已发射”的证据）。已发射、已消耗、已损失的弹一律
        拒绝 —— 不复活。想定 7527 的三枚弹初态就是 `unassigned`（`car=Obj_0`）。
        """
        c = self.units.get(carrier_id)
        m = self.units.get(missile_id)
        block = self.missile_deploy_block(carrier_id, missile_id)
        if block:
            return False, block
        m.detail["carrier"] = c.id
        c.detail.setdefault("carried", []).append(m.id)
        # 与人员上车同理：车内读不到兵力与阵营（平台实测），编制留在 `establishment`。
        m.establishment = m.establishment or m.blood
        m.blood = 0
        m.color = ""
        m.detail["missile_state"] = "carried"
        self.record("%s 部署巡飞弹 %s（手册 4.10(3)）" % (c.army or c.name, m.id))
        return True, ""

    def launch_missile(self, carrier_id, missile_id):
        """手册 4.10(3)(4)：由所属车辆发射；该车**同时只能有一枚在飞**。

        与原实现的三处差别（issue #1）：

        1. **不再用“一局一枚”的永久标记** —— 占用由 `in_flight_missile()` 判定，
           弹一被消耗 / 自毁 / 载车被毁即释放（原实现的 `launched_missile` 永不清除）。
        2. **不再用 `alive()` 拦车内弹** —— 车内 `blood=0` 是“车内读不到兵力”，不是
           “已损失”；“已损失不能复活”改由**显式生命周期状态**判定。
        3. **必须 `self.started`** —— 手册 4.10(3)：开始后才能发射（原实现在开局前也放行）。
        """
        c = self.units.get(carrier_id)
        m = self.units.get(missile_id)
        if not c or not m:
            return False, "单位不存在"
        if not m.is_loitering_missile():
            return False, "该棋子不是巡飞弹"
        if not self.started:
            return False, "对抗开始前不能发射巡飞弹（手册 4.10(3)）"
        state = self.missile_state(m)
        if state in ("spent", "lost"):
            return False, "巡飞弹已消耗/已损失（状态 %s），不能再次发射" % state
        if state != "carried":
            return False, "巡飞弹未部署上车（手册 4.10(3)）"
        if m.detail.get("carrier") != c.id:
            return False, "巡飞弹部署在别的载车上（手册 4.10(3)）"
        if not c.alive():
            return False, "载车已被消灭"
        if not c.color:
            return False, "载车没有阵营"
        if self.in_flight_missile(c.id) is not None:
            return False, "该车已有一枚巡飞弹在飞（手册 4.10(4)）"
        # 发射：把弹从载车的载员表里**移出**（不得重复卸载/重复发射同一枚）。
        m.detail.pop("carrier", None)
        carried = list(c.detail.get("carried") or [])
        if m.id in carried:
            carried.remove(m.id)
        c.detail["carried"] = carried
        # 发射后弹体是图上的独立单位：恢复阵营与编制兵力（与“下车”同理）。
        m.color = c.color
        if m.blood <= 0:
            establishment = m.establishment or carried_establishment(m.data)
            if establishment is None:
                # 编制未知：不臆造，标出来让计分显式报出。
                m.detail["establishment_unverified"] = True
                establishment = 0
            m.blood = establishment
            m.establishment = m.blood
        m.units = max(1, m.blood) if m.typeid in (1, 2) else 1
        m.pos = c.pos
        m.pos2 = c.pos
        m.detail["missile_state"] = "loitering"
        m.detail["launcher"] = c.id
        m.detail["launch_second"] = self.second
        # 时间字段来源：客户端用 ObjAjax 下发的 `radar` 算巡飞计时；这里原样留存。
        m.detail["launch_radar"] = m.radar or ""
        m.detail["loiter_seconds"] = self.loiter_sec
        m.detail["loiter_seconds_source"] = self.loiter_sec_source
        m.detail["loiter_until"] = self.second + self.loiter_sec
        c.detail["launched_missile"] = m.id
        c.detail["stop"] = 1              # 平台：载车在飞占用（客户端读 `stop` 决定能否再发）
        self.record("%s 发射巡飞弹 %s（巡飞上限 %s 秒，取值来源 %s）"
                    % (c.army or c.name, m.id, self.loiter_sec,
                       self.loiter_sec_source))
        return True, ""

    def _terminate_missile(self, m, why):
        """在飞弹终止（超时自毁 / 载车被毁）：置 `lost`、清动作并释放载车占用。"""
        m.blood = 0
        m.detail["missile_state"] = "lost"
        m.detail.pop("loiter_until", None)
        self._clear_missile_actions(m)
        self.release_launcher(m.detail.get("launcher"))
        self.record("巡飞弹 %s 终止：%s" % (m.id, why))

    def _consume_missile(self, m):
        """攻击完成后弹体消失（负责人 2026-10-16 确认），并释放载车占用。

        只由**已执行**的攻击调用：所有校验拒绝都在 `attack()` 里提前返回，走不到这里，
        所以被拒绝的（没有实际执行的）攻击不会误消耗弹体（issue #1 第 4 条）。
        """
        m.blood = 0
        m.detail["missile_state"] = "spent"
        m.detail["consumed"] = True
        m.detail.pop("loiter_until", None)
        self._clear_missile_actions(m)
        self.release_launcher(m.detail.get("launcher"))
        self.record("巡飞弹 %s 攻击完成，弹体消耗（负责人确认：攻击后消失）" % m.id)

    @staticmethod
    def _clear_missile_actions(m):
        """消耗/损失后不得继续移动、侦察或重复攻击：清掉机动路径与持续打击。"""
        for key in ("path", "path_idx", "move_until", "target", "target_weapon"):
            m.detail.pop(key, None)
        m.pos2 = m.pos

    def _expire_missiles(self, now):
        """手册 4.10(5)：巡飞超时自毁；发射车被摧毁则自毁。

        两个终止分支都**释放载车占用**（`release_launcher`）—— 原实现只把弹血量清零，
        载车的 `launched_missile` 永久留着，于是“第二枚”永远发不出去。
        """
        for u in self.units.values():
            if not u.is_loitering_missile() or self.missile_state(u) != "loitering":
                continue
            deadline = u.detail.get("loiter_until")
            if deadline and deadline <= now:
                self._terminate_missile(u, "巡飞超时自毁（手册 4.10(5)）")
                continue
            lc = self.units.get(u.detail.get("launcher"))
            if lc is None or not lc.alive():
                self._terminate_missile(u, "发射车被摧毁（手册 4.10(5)）")
        # 载车占用与实际在飞弹保持一致（弹已被消耗/损失 ⇒ 立即释放，不留永久锁）。
        for c in self.units.values():
            if c.detail.get("launched_missile") and self.in_flight_missile(c.id) is None:
                self.release_launcher(c.id)

    @property
    def time1(self):
        return 1 if self.started else ""

    # ---------------- 地形 / 通行 ----------------
    # 格的 `cond` → 地形类别（纯函数，只依赖一个标量）。
    # ★ 不要按 mapid 缓存整个格字典：`hexes` 是公开可变字典（测试与核查脚本会
    #   原地改 `step`/`elevation` 来验证规则），按 mapid 缓存会拿到旧值。
    #   这里只去掉重复的分支判类，取值与直算**完全相同**。
    _TERRAIN_BY_COND = {7: "居民地", 6: "过河", 8: "丛林地", 9: "丛林地"}

    def hex(self, mapid):
        return self.hexes.get(mapid if type(mapid) is int else int(mapid))

    def elevation(self, mapid):
        h = self.hex(mapid)
        return int(h["elevation"]) if h else 0

    def unit_height(self, u):
        """单位用于通视计算的等效高程（米）= 所在格高程 + 滞空高度。

        手册 4.8(2)：无人机滞空按所在格高程 +500/200/20 米计算通视；
        手册 4.10(1)：巡飞弹按所在格高程 +200 米。数据里即单位属性 `e2`。
        """
        h = self.elevation(u.pos)
        return h + (u.alt or 0) if u.typeid == 3 else h

    def terrain_kind(self, mapid):
        """粗略地形类别（由 cond 推断）。"""
        h = self.hex(mapid)
        if not h:
            return None
        return self._TERRAIN_BY_COND.get(int(h.get("cond") or 0))

    def passable(self, mapid, unit):
        h = self.hex(mapid)
        if h is None:
            return False, "地图外"
        step = int(h.get("step") or 0)
        if step == 0:
            return False, "禁行区"
        if unit.typeid == 2:
            terr = self._TERRAIN_BY_COND.get(int(h.get("cond") or 0))
            if terr == "过河":
                return False, "车辆不可跨越水系"
            if step == -1 and unit.c0 != 2:
                return False, "车辆禁行"
        return True, ""

    def stack_index(self):
        """`(color, pos) → 地面单位数`（typeid 1/2、存活、未上车）。

        ★ 只读快照，**不缓存在 Game 上**：调用方（如枚举的机动 BFS）自己一次成型、
        自己保证“期间没有位移”，所以不可能拿到陈旧值。等价于 `step_seconds`
        每次现场扫全部单位数一遍，只是从 O(单位数) 降成 O(1)。
        """
        idx = {}
        for u in self.units.values():
            if u.blood > 0 and u.typeid in (1, 2) and not u.detail.get("carrier"):
                key = (u.color, u.pos)
                idx[key] = idx.get(key, 0) + 1
        return idx

    def step_seconds(self, unit, from_id, to_id, stack=None):
        """返回进入 to_id 一格所需秒数（客户端会 ×1000 作为动画时长）。

        `stack` 可选：调用方预先用 `stack_index()` 做好的位置快照；不传则现场数。
        """
        if not unit.alive() or unit.detail.get("carrier"):
            return 0, "单位不能独立移动"
        if self.hex(from_id) is None or G.distance(from_id, to_id) != 1:
            return 0, "必须移动到相邻格"
        ok, why = self.passable(to_id, unit)
        if not ok:
            return 0, why
        hto = self.hex(to_id)          # `passable` 已确认非空，直接取派生字段
        to_ele = int(hto["elevation"])
        terr = self._TERRAIN_BY_COND.get(int(hto.get("cond") or 0))
        to_note = hto.get("note") or ""
        base = {1: 60.0, 2: 12.0, 3: 6.0, 4: 999.0}.get(unit.typeid, 60.0)
        # 手册 4.6.3(1)：人员被压制后不能机动；需等压制解除后才能继续（4.1.1(6)）。
        if unit.typeid in (1, 2) and unit.stunned(self.second):
            return 0, "被压制不能机动"
        if unit.typeid == 1:
            if terr == "过河":
                base *= 2.0            # 手册 4.1.1(1) 步兵过河降为 1/2
            dh = abs(to_ele - self.elevation(from_id))
            if dh // 20 >= 4:
                base *= 2.0            # 手册 4.1.1(3) 坡度等级 ≥4 → 1/2 速
            if unit.ste == 4:          # 手册 4.1.1(4) 冲锋 = 越野速度 4 倍
                if unit.fatigued(self.second):
                    return 0, "疲劳状态下不能继续冲锋（手册 4.1.1(4)）"
                base /= 4.0
        elif unit.typeid == 2:
            if to_note in self.ROAD_NOTES:
                # 手册 4.1.2(4)：沿道路行驶，速度不受地形和高差影响
                #（同时豁免 4.1.2(3) 的高差减速与 >100 米禁行）
                pass
            else:
                if terr == "丛林地":
                    base *= 2.0            # 手册 4.1.2(1)
                elif terr == "居民地":
                    base *= 3.0
                dh = abs(to_ele - self.elevation(from_id))
                if dh > 100:
                    return 0, "高差>100 米，车辆无法进入并通过（手册 4.1.2(3)）"
                lvl = dh // 20
                if lvl >= 1:
                    base *= (lvl + 1)
        # 堆叠限制：同格本方地面单位不超过 4
        # 手册 4.2：同格本方地面单位不可超过 4 个（已上车的步兵不占格位）
        # ★ 判据顺序与提前退出只是优化：先比 `pos`（绝大多数单位直接落空），
        #   数到 4 就够了 —— 与 `sum(...) >= 4` 的结论完全一致。
        if stack is None:
            n = 0
            for u in self.units.values():
                if (u.pos == to_id and u.blood > 0 and u.color == unit.color
                        and u.typeid in (1, 2) and u.id != unit.id
                        and not u.detail.get("carrier")):
                    n += 1
                    if n >= 4:
                        return 0, "违反堆叠规则"
            return base, ""
        n = stack.get((unit.color, to_id), 0)
        if (unit.pos == to_id and unit.blood > 0 and unit.typeid in (1, 2)
                and not unit.detail.get("carrier")):
            n -= 1                      # 索引里包含自己，旧口径是 `u.id != unit.id`
        if n >= 4:
            return 0, "违反堆叠规则"
        return base, ""

    def commit_step(self, unit, from_id, to_id):
        """Validate a step and commit its costs; the caller updates positions.

        Unlike step_seconds(), this is a mutating operation. Call once for
        each accepted step, before updating the unit's position.
        """
        seconds, why = self.step_seconds(unit, from_id, to_id)
        if seconds > 0 and unit.typeid == 1 and unit.ste == 4:
            self._charge_meters(unit, self.CHARGE_FATIGUE_M)
        return seconds, why

    CHARGE_FATIGUE_M = 200      # 手册 4.1.1(4)：每冲锋 200 米转为疲劳状态
    # 手册 4.3(2)：空中单位和炮兵不能执行夺控任务
    NO_CAPTURE_HINT = ("自行火炮", "榴弹炮", "加农炮", "火箭炮", "迫击炮", "炮兵")
    # 手册 4.11(3)：同格交战的自动攻击间隔为正常攻击间隔的 1/3
    ATTACK_INTERVAL_SEC = 10
    ENGAGE_INTERVAL_SEC = max(1, ATTACK_INTERVAL_SEC // 3)
    # 手册 4.1.2(4)：沿道路以机动状态行驶时，速度不受地形和高差影响。
    # 想定 7527 的道路标记落在格的 `note` 字段（观测到取值「主要」「次要」）。
    ROAD_NOTES = ("主要", "次要")
    @classmethod
    def loiter_seconds(cls, room_speed=DEFAULT_ROOM_SPEED,
                       room_units=LOITER_ROOM_UNITS):
        """客户端 `FlyTime = parseInt(1200 / RoomSpeed)` → 巡飞超时上限（秒）。

        ⚠️ **未确证**：`radar` / `ServerTime` 的尺度与 `RoomSecond`（本引擎 `Game.second`）
        是否完全一致**没有实测样本**；本函数只把客户端公式搬到训练时钟上，并把取值的
        来源逐弹记进 `detail["loiter_seconds_source"]`，不宣称已与平台对账。
        """
        if isinstance(room_speed, bool) or not isinstance(room_speed, (int, float)) \
                or room_speed <= 0:
            raise ValueError("room_speed must be a positive number")
        return max(1, int(room_units // room_speed))

    def _charge_meters(self, unit, meters):
        """手册 4.1.1(4)：人员每冲锋 200 米转为疲劳状态。

        一格按 200 米计，因此冲锋一格即触发；疲劳后会阻止继续冲锋（见 step_seconds）。
        移除时机见 4.1.1(5)：停止机动并等待一个疲劳移除时长。
        """
        acc = unit.detail.get("charged_m", 0) + meters
        if acc >= self.CHARGE_FATIGUE_M:
            acc -= self.CHARGE_FATIGUE_M
            unit.fatigued_until = self.second + FATIGUE_SEC
            self.record("%s 冲锋达到 %d 米，转入疲劳"
                        % (unit.army or unit.name, self.CHARGE_FATIGUE_M))
        unit.detail["charged_m"] = acc

    # ---------------- 通视 / 观察 ----------------
    def _los_blocks(self, mapid, endpoint_ids):
        """连线经过该格时，其用于阻挡的等效高程。"""
        e = self.elevation(mapid)
        if mapid in endpoint_ids:
            return e          # 端点所在格的居民地/丛林地不修正
        terr = self.terrain_kind(mapid)
        if terr in ("居民地", "丛林地"):
            e += 1
        return e

    def line_hexes(self, a, b):
        """a→b 的格序列（线性插值取整）。"""
        ax, ay = G.pos_of(a)
        bx, by = G.pos_of(b)
        n = max(abs(ax - bx), abs(ay - by))
        steps = max(1, int(n / (G.HEX_W / 2)))
        # 用立方坐标插值更稳
        aq, ar, as_ = G.to_cube(a)
        bq, br, bs = G.to_cube(b)
        d = max(abs(aq - bq), abs(ar - br), abs(as_ - bs))
        out = []
        for i in range(d + 1):
            t = i / d if d else 0
            q = aq + (bq - aq) * t
            r = ar + (br - ar) * t
            out.append(self._cube_round(q, r))
        return out

    @staticmethod
    def _cube_round(q, r):
        s = -q - r
        rq, rr, rs = round(q), round(r), round(s)
        dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
        if dq > dr and dq > ds:
            rq = -rr - rs
        elif dr > ds:
            rr = -rq - rs
        row = rr
        col = rq + (rr - (rr & 1)) // 2
        return G.make_mapid(row, col)

    def los(self, a, b, h_obs=None, h_tar=None):
        """通视判定（手册 4.4：高看低，高者所在格高程 +1 个等级）。"""
        if a == b:
            return True, []
        pa = G.distance(a, b)
        if pa <= 1:
            return True, []          # 相邻格总是通视
        ea = self.elevation(a)
        eb = self.elevation(b)
        if h_obs is None:
            h_obs, h_tar = (ea, eb) if ea >= eb else (eb, ea)
        else:
            if h_obs >= h_tar:
                ea, eb = h_obs, h_tar
                obs_is_a = True
            else:
                ea, eb = h_tar, h_obs
                obs_is_a = False
        # 观察点所在格高程 +1 个等级（20m）
        ea += 20
        path = self.line_hexes(a, b)
        blockers = []
        for i, hid in enumerate(path):
            if hid in (a, b):
                continue
            t = i / (len(path) - 1) if len(path) > 1 else 0
            line_e = ea + (eb - ea) * t
            be = self._los_blocks(hid, {a, b})
            if be > line_e:
                blockers.append(hid)
        return (len(blockers) == 0), blockers

    def observe_range(self, obs, tgt):
        """观察者对被观察者的有效观测距离（格）。"""
        eq = EQUIPMENT.get(obs.data, None)
        if obs.typeid == 3:
            base = 2 if obs.name in ("旋翼无人机", "自爆无人机", "巡飞弹") else 2
            rng = base
        else:
            rng = (eq[4][1] if eq else 25) if tgt.typeid == 2 else (eq[4][0] if eq else 10)
            if tgt.typeid == 3:
                rng = (eq[4][2] if eq else 1)
        if obs.typeid == 3:
            rng = 2 if tgt.typeid != 3 else 0
        terr = self.terrain_kind(tgt.pos)
        if terr in ("居民地", "丛林地"):
            rng *= 0.5
        if tgt.ste == 2 or tgt.detail.get("hidden"):
            # 手册 4.5(1)：掩蔽状态下被观察距离减半。
            # 例外（手册 4.8(1)/4.10(2)）：无人机、巡飞弹对**掩蔽中的车辆**侦察距离
            # 不受影响，只有掩蔽中的**人员**才减半。
            if not (obs.typeid == 3 and tgt.typeid == 2):
                rng *= 0.5
        return rng

    def can_observe(self, obs_id, tgt_id):
        o = self.units.get(obs_id)
        t = self.units.get(tgt_id)
        if not o or not t or not o.alive() or not t.alive():
            return False
        if o.out_of_control():
            return False                  # 手册 4.9(3)：失控后不能侦察
        d = G.distance(o.pos, t.pos)
        if d > self.observe_range(o, t):
            return False
        ok, _ = self.los(o.pos, t.pos, h_obs=self.unit_height(o),
                         h_tar=self.unit_height(t))   # 手册 4.8(2)/4.10(1) 滞空高度
        return ok

    # ---------------- 指令 ----------------
    def new_command(self, data):
        self.com_seq += 1
        cid = self.com_seq
        self.orders[cid] = {"data": data, "ts": self.now_str(), "done": False}
        return cid

    def now_str(self):
        return time.strftime("%Y/%m/%d %H:%M:%S")

    def apply_city_capture(self, unit):
        """夺控：到达夺控点格且该格及相邻 6 格无敌方地面单位。

        手册 4.3(2)：空中单位和炮兵不能执行夺控任务。
        """
        if (not unit.alive() or not unit.color or unit.detail.get("carrier")
                or unit.blood < CAPTURE_MIN_BLOOD or unit.typeid == 3):
            return None
        if any(k in (unit.name or "") for k in self.NO_CAPTURE_HINT):
            return None
        for c in self.cities:
            if c["pos"] != unit.pos:
                continue
            zone = [c["pos"]] + G.neighbors(c["pos"])
            enemy = [u for u in self.units.values()
                     if u.alive() and u.color and unit.color and u.color != unit.color
                     and u.typeid in (1, 2) and u.pos in zone
                     and not u.detail.get("carrier")]
            if enemy:
                return None
            c["flag"] = unit.color or c.get("flag")
            c["ico"] = unit.color or c.get("ico")
            self.record("%s 夺占 %s" % (unit.army or unit.name, c.get("name")))
            return c
        return None

    def record(self, text):
        self.log.append("[%s] %s" % (self._fmt_sec(self.second), text))
        if len(self.log) > 400:
            self.log = self.log[-400:]

    @staticmethod
    def _fmt_sec(s):
        return "%02d:%02d" % (s // 60, s % 60)

    # ---------------- 直瞄射击 ----------------
    def attack(self, obj_id, tar_id, weapon_id=0, com_id=0, dist=None):
        a = self.units.get(obj_id)
        t = self.units.get(tar_id)
        if not a or not t:
            return None, "棋子不存在"
        if not a.alive():
            return None, "射击单位已被消灭"
        if not t.alive():
            return None, "目标已被消灭"
        if a.stunned(self.second):
            a.keep = 1
            return None, "压制状态下不能射击"
        if a.out_of_control():
            # 手册 4.9(3)：无人战车失去依托后不能射击/侦察/校射
            return None, "无人战车失去依托，不能射击（手册 4.9(3)）"
        if a.is_loitering_missile() and not a.detail.get("launcher"):
            # 手册 4.10(3)：未部署上车的巡飞弹不能发射
            return None, "巡飞弹未部署上车，不能发射（手册 4.10(3)）"
        # 同格交战
        if a.color == t.color:
            return None, "不能攻击己方"

        units = max(1, a.blood if a.typeid in (1, 2) else 1)
        wid = weapon_id if weapon_id in a.wz else (a.wz[0] if a.wz else 0)
        w = WEAPONS.get(wid)
        if not w:
            return None, "无可用武器"
        wname, ammo_type, wrange = w

        actual_dist = G.distance(a.pos, t.pos)
        if dist is not None and (isinstance(dist, bool) or not isinstance(dist, int)
                                 or dist != actual_dist):
            return None, "距离与棋盘坐标不符"
        dist = actual_dist
        if dist > wrange:
            return None, "超出射程(%d格>%d格)" % (dist, wrange)

        # 通视与观察
        ok, blk = self.los(a.pos, t.pos)
        if not ok:
            return None, "不通视(%s-%s)" % (G.hexid(a.pos), G.hexid(t.pos))
        if not self.can_observe(obj_id, tar_id):
            return None, "不可观察(%s-%s)" % (G.hexid(a.pos), G.hexid(t.pos))

        # 弹药
        fld = AMMO_FIELD.get(ammo_type)
        avail = a.ammo.get(fld, 0) if fld else None
        if fld and avail is not None and avail <= 0:
            return None, "%s 弹药不足" % ammo_type

        dh = self.elevation(a.pos) - self.elevation(t.pos)
        terr = self.terrain_kind(t.pos)
        # 叠加状态
        stacked = sum(1 for u in self.units.values()
                      if u.alive() and u.color == t.color and u.pos == t.pos
                      and u.typeid in (1, 2) and u.id != t.id) > 0
        r = crt.direct_fire(
            shooter=dict(typeid=a.typeid, units=units, moving=a.moving(self.second),
                         stunned=a.stunned(self.second)),
            target=dict(typeid=t.typeid, armor=t.armor, hidden=(t.ste == 2),
                        moving=t.moving(self.second), stacked=stacked),
            weapon_name=wname, dist=dist, dh_m=dh, terrain=terr, rng=self.rng,
        )
        if r["result"] == "NO_WEAPON":
            return None, "；".join(r["detail"])

        if fld:
            a.ammo[fld] = max(0, a.ammo.get(fld, 0) - 1)

        # 应用战果
        msg = []
        kills = r["kills"]
        # 手册 4.6.3(2) 需要知道目标"原本是否已处于被压制"
        was_stunned = t.stunned(self.second)
        if kills > 0:
            t.blood -= kills
            if t.blood < 0:
                t.blood = 0
            msg.append("消灭 %d" % kills)
            # 车辆被消灭，车内人员同损（手册 4.6.3(3)）
            if t.typeid == 2:
                for s in self.units.values():
                    if s.alive() and s.id != t.id and s.pos == t.pos and s.typeid == 1 \
                            and s.color == t.color and s.blood > 0 \
                            and s.detail.get("carrier") == t.id:
                        s.blood = max(0, s.blood - kills)
        if r["suppressed"]:
            t.keep = 1
            t.stunned_until = self.second + SUPPRESS_SEC
            msg.append("压制")
        # 手册 4.6.3(2)：被压制的人员再次被裁决压制，则损失 1 个兵力，剩余兵力仍
        # 保持被压制。该条针对的是"再次压制"（本次未产生新伤亡）；若本次有伤亡，
        # 按 4.6.1 的互斥口径只计伤亡，不叠加本条，否则会重复计数。
        if t.typeid == 1 and was_stunned and t.blood > 0 \
                and r["suppressed"] and kills == 0:
            t.blood -= 1
            t.keep = 1
            t.stunned_until = self.second + SUPPRESS_SEC
            msg.append("再次被压制，额外损失 1 个班（手册 4.6.3(2)）")
        if t.blood <= 0:
            msg.append("已被消灭")
        a.att = 1
        self.fires[a.id] = {"x": 0, "y": 0, "type": 1, "sec": 2}
        self.record("直瞄：%s → %s（%s）%s" % (a.army or a.name, t.army or t.name,
                                              "；".join(r["detail"]), "，".join(msg) or "无效"))
        self.dirty = True
        if a.is_loitering_missile():
            # 负责人 2026-10-16 确认：攻击后巡飞弹消失 —— 不能继续移动、侦察或重复攻击，
            # 同时解除发射车的在飞占用。放在这里而不是开头：上面每一条校验拒绝都是
            # **提前 return**，所以"被拒绝、没有实际执行的攻击不会误消耗弹体"。
            self._consume_missile(a)
        return r, None

    # ---------------- 持续打击 / 集火射击 ----------------
    def set_engagement(self, obj_id, target_id, weapon_id=None):
        """手册 3.14：为棋子设定打击目标后，该棋子**持续对其打击直到目标被消灭**，
        无需重复下达指令。

        选定武器必须一并记住：否则后续每次自动开火都会退回 wz[0]
        （例如坦克的 wz[0] 是高射机枪，打装甲几乎没战果）。
        """
        u = self.units.get(obj_id)
        t = self.units.get(target_id)
        if not u or not t or not u.alive() or not t.alive():
            return None, "棋子不存在"
        if u.color and t.color and u.color == t.color:
            return None, "不能攻击己方"
        u.detail["target"] = target_id
        u.detail["target_weapon"] = int(weapon_id or 0)
        self.record("%s 设定打击目标：%s（将持续打击至其被消灭）"
                    % (u.army or u.name, t.army or t.name))
        return target_id, None

    def clear_engagement(self, obj_id):
        """取消持续打击。"""
        u = self.units.get(obj_id)
        if not u:
            return None, "棋子不存在"
        u.detail.pop("target_weapon", None)
        return u.detail.pop("target", None), None

    def resolve_engagements(self):
        """手册 3.14：已设定打击目标的棋子自动持续打击，按攻击间隔结算。

        - 目标被消灭 → 自动解除，本次不再射击；
        - 目标暂时超出射程/不通视/己方被压制 → 本次不射击，但**目标仍然保留**，
          条件恢复后自动继续（这才符合“持续打击直到目标被消灭”）。
        """
        if self.second - getattr(self, "_last_volley", -10 ** 9) < self.ATTACK_INTERVAL_SEC:
            return []
        self._last_volley = self.second
        out = []
        for u in list(self.units.values()):
            if not u.alive() or u.detail.get("carrier"):
                continue
            tid = u.detail.get("target")
            if not tid:
                continue
            t = self.units.get(tid)
            if not t or not t.alive():
                u.detail.pop("target", None)
                u.detail.pop("target_weapon", None)
                out.append((u.id, tid, "目标已被消灭，解除打击"))
                continue
            r, err = self.attack(u.id, t.id,
                                 weapon_id=int(u.detail.get("target_weapon") or 0))
            out.append((u.id, tid, err or r["result"]))
        if out:
            self.record("持续打击结算：%d 个单位" % len(out))
        return out

    def area_fire(self, unit_ids, target_hex, weapon_id=None):
        """手册 3.16 集火射击：多个棋子对**某个六角格**实施集火。

        与直瞷射击"跟踪目标并持续射击"不同：集火射击指定的目标是**格坐标**，
        射击时目标若已离开该格，则本次集火射击**失败**，不跟踪、不持续。
        weapon_id 为 None 时由 attack() 自行选武器。
        """
        shooters = [self.units.get(i) for i in unit_ids]
        shooters = [u for u in shooters if u and u.alive()]
        if not shooters:
            return None, "没有可用的射击单位"
        if not self.hex(target_hex):
            return None, "目标格不存在"
        color = shooters[0].color
        targets = [u for u in self.units.values()
                   if u.alive() and u.pos == target_hex and u.color and u.color != color
                   and not u.detail.get("carrier")]
        if not targets:
            self.record("集火射击：%s 已无目标，本次射击失败（手册 3.16）"
                        % G.hexid(target_hex))
            return {"result": "MISS", "shots": [],
                    "detail": ["目标已离开该格，集火射击失败"]}, None
        shots = []
        for s in shooters:
            for t in targets:
                r, err = self.attack(s.id, t.id,
                                     weapon_id=int(weapon_id) if weapon_id else 0)
                shots.append({"unit": s.id, "target": t.id,
                              "result": None if err else r["result"],
                              "error": err})
        self.record("集火射击：%d 个单位 → %s（%d 次裁决）"
                    % (len(shooters), G.hexid(target_hex), len(shots)))
        return {"result": "AREA", "shots": shots}, None

    # ---------------- 手雷 / 同格交战 ----------------
    def grenade(self, obj_id, target_hex):
        """手册 4.7 手雷：对目标格内**所有人员目标（含己方）**各裁决一次。

        - 不能摧毁空中单位（typeid 3）和掩体（typeid 4）；
        - 裁决有效则损失 1 个班，剩余兵力被压制（4.7(3)）；
        - 若目标本已被压制，再次裁决有效仍损失 1 个班（4.7(4) "二次压制"）。
        """
        a = self.units.get(obj_id)
        if not a or not a.alive():
            return None, "投掷单位不存在"
        if a.typeid != 1:
            return None, "只有人员单位可投掷手雷（手册 3.15）"
        if a.stunned(self.second):
            return None, "压制状态下不能射击（手册 4.6.3(1)）"
        if 144 not in a.wz:
            return None, "该单位没有手雷"
        if not self.hex(target_hex):
            return None, "目标格不存在"
        if G.distance(a.pos, target_hex) > 1:
            return None, "手雷投掷范围通常为相邻格（手册 3.15）"
        if a.ammo.get("a5", 0) <= 0:
            return None, "弹药不足"
        a.ammo["a5"] = max(0, a.ammo.get("a5", 0) - 1)
        if a.ammo.get("a5", 0) <= 0:
            a.wz = [x for x in a.wz if x != 144]

        terr = self.terrain_kind(target_hex)
        out = []
        for u in list(self.units.values()):
            if not u.alive() or u.pos != target_hex:
                continue
            if u.typeid == 3:
                out.append((u, "空中单位不能被手雷摧毁（手册 4.7(2)）"))
                continue
            if u.typeid == 4:
                out.append((u, "掩体不能被手雷摧毁（手册 4.7(2)）"))
                continue
            if u.typeid != 1:
                out.append((u, "手雷只对人员单位裁决"))
                continue
            roll = crt.roll2d6(self.rng)
            mod = crt.situational_mod(
                terrain=terr,
                target_hidden=(u.ste == 2 or u.detail.get("hidden")),
            )
            corrected = roll + mod
            v = crt.result_grenade(corrected)
            if v and v >= 1:
                second = u.stunned(self.second)
                u.blood = max(0, u.blood - 1)
                u.keep = 1
                u.stunned_until = self.second + SUPPRESS_SEC
                out.append((u, "损失 1 个班%s，剩余被压制（手册 4.7(%d)）"
                            % ("（二次压制）" if second else "", 4 if second else 3)))
            else:
                out.append((u, "无效"))
            if u.blood <= 0:
                out.append((u, "已被消灭"))

        self.record("手雷：%s → %s（%s）" % (
            a.army or a.name, G.hexid(target_hex), "；".join(
                "%s %s" % (u.army or u.name, m) for u, m in out) or "格内无目标"))
        self.dirty = True
        return out, None

    def same_hex_engage(self):
        """手册 4.11 同格交战。

        一方地面棋子进入另一方地面棋子所在格即自动触发；格内棋子自动交战、
        被攻击者自动还击，攻击间隔为正常攻击间隔的 1/3。这里实现为
        "每次推进时钟最多结算一轮"，攻击顺序按 uid（近似"进入该格的先后"）。
        手册 4.11(5)：AI 控制的棋子不发起同格交战。
        """
        if self.second - getattr(self, "_last_engage", -10 ** 9) < self.ENGAGE_INTERVAL_SEC:
            return []
        self._last_engage = self.second

        by_hex = {}
        for u in self.units.values():
            # 已上车的步兵随车机动，不单独参与同格交战
            if u.alive() and u.typeid in (1, 2) and not u.detail.get("carrier"):
                by_hex.setdefault(u.pos, []).append(u)

        rounds = []
        for pos, us in by_hex.items():
            if len({u.color for u in us if u.color}) < 2:
                continue
            order = sorted(us, key=lambda x: (x.uid or 0, x.id))
            for atk in order:
                if not atk.alive() or atk.ai:
                    continue
                for dfn in order:
                    if dfn is atk or not dfn.alive() or dfn.color == atk.color:
                        continue
                    r, err = self.attack(atk.id, dfn.id, dist=0)
                    rounds.append((atk.id, dfn.id, err or r.get("result")))
                    break
        if rounds:
            self.record("同格交战：%d 次裁决（间隔 %d 秒，手册 4.11(3)）"
                        % (len(rounds), self.ENGAGE_INTERVAL_SEC))
        return rounds

    # ---------------- ObjAjax 快照 ----------------
    def obj_state(self, u):
        """客户端 ObjAjax 所需字段（键名与线上一致）。"""
        now = self.second
        st = {
            "uid": str(u.uid), "color": u.color, "pos": str(u.pos), "pos2": str(u.pos2),
            "rot": str(u.rot), "son": "0", "sonarr": "", "blood": str(u.blood),
            "keep": str(1 if u.stunned(now) else 0), "see": str(u.see),
            "ste": 4 if u.ste == 4 else 0, "pass": str(u.pass_),
            "score": u.score, "runed": 0, "att": u.att, "datetime": "",
            "radar": u.radar or "", "aim": 0,
            "a0": 0, "a2": 0, "c0": u.c0, "c3": u.c3, "d0": 0,
            "f1": 0, "f2": 0, "e2": u.detail.get("e2", 0),
            "f3": 0, "r3": 0, "r4": 0, "ai": u.ai, "user": str(u.uid or ""),
        }
        return st
