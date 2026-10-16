# -*- coding: utf-8 -*-
r"""实时（1 Hz）双席对局接口 —— 与正式规则同粒度。

## 解决什么问题

`TwoPlayerEpisode` 是**回合制抽象**：一次 `step` = 一方的一个单位的一条指令，
时钟按这条指令自己的时长整段跳。实测后果（`tools/hex_time_acceptance.py`）：

* 1200 秒的一局，每方只拿到 **15** 次决策（正赛里应是每秒一次）；
* 一条 60 秒的机动把时钟从 0 直接推到 60，**期间对手没有任何决策点**；
* 自动裁决（3.14 持续打击 / 4.11 同格交战 / 压制解除）的次数取决于"把时间切成几段"
  —— 同一段 60 秒，拆成 1 秒一段能打出 **41** 发，一次跳 60 秒只打出 **3** 发。

本模块提供 **`RealtimeEpisode`**：

| | 语义 |
|---|---|
| 时钟 | **1 Hz 逐秒推进**（`Game.tick_second()`） |
| 决策机会 | **每秒、双方各一次**；一局 1200 秒 ⇒ 每方 1200 次 |
| 持续指令 | 机动 = 登记路径（`Game.begin_move`），**在统一时间轴上按秒推进**；持续打击 = 3.14 自动结算 |
| 并行 | 所有单位的指令**同时**执行；某单位的机动不阻塞其他单位 |
| `keep` vs `stop` | `{"type":"keep"}` = **保持现有指令**；`{"type":"stop","unit":X}` = **停止该单位的行动** |
| 单位占用 | 射击/夺控/装卸/掩蔽会占用该单位若干秒（`busy_until`），期间不能再下令 |
| 观测 | 每个单位带 `order`（在执行什么、还剩几秒）；顶层带 `time_left` |

## 与旧接口的关系

`TwoPlayerEpisode` 原样保留（不破坏既有测试与基线），本模块是**新接口**。
两者共用同一套 `engine/`，所以规则只有一份。

## 训器侧接入（2026-10-17）

`env/neural_train.py` 新增 `interface="realtime"`（默认仍 `serial`）：每秒双方各一次
决策、学习目标 $R = S_{\text{终}} - S_{\text{开局}}$；配置 `train/configs/ppo_realtime.json`。
成本实测：真实地图上**一次决策 ≈ 22 ms**（策略前向+特征）+ 环境 6 ms ⇒ 单局 1200 次决策
≈ 107 s 墙钟。

## 实测修过的坑（2026-10-17）

`_enumerate` 以前把“可见敌人”重建为 `Unit({id,name,typeid,color,pos,hexid,blood,suppressed})`
—— 只有 8 个公共字段，**丢掉 `hide`（掩蔽）/`ste`/`e2`（滞空高度）**。这两个字段正是
`observe_range`（掩蔽距离减半）与 `unit_height`（LOS 高度）的输入 ⇒ 投影里能打的攻击，
到权威状态上会被 `can_observe` 拒（`射击失败：不可观察`）。随机策略一撞即崩。
现在可见敌人直接用**权威对象本身**（与己方同源），迷雾不变（只放本 tick 可见的敌人）。

## 奖励

本模块**只负责环境**：`status()` 给出 `final_scores` 与 `opening_scores`。
按用户的训练目标 $\mathbb E[S_R + S_B]$（红蓝各一局、以**本方两局得分和**为主指标），
奖励建议在**训练器侧**做：

$$r_t = \\frac{S_{\\text{本方},t+1} - S_{\\text{本方},t}}{c},\\qquad
R = S_{\\text{终}} - S_{\\text{开局}}$$

`opening_scores[side]` 就是 $S_{\\text{开局}}$（7527 为 蓝 234 / 红 354）。
"""
import copy
import json
import math

from engine.game import Unit, DEPLOY_WINDOWS
from engine import scoring as _scoring
from .hexwar_env import HexwarEnv, own_units_for_projection
from .match_eval import SIDES, _SerialSession, _json_copy, _positive_int

reward_value = _scoring.reward_value

# 非机动动作对单位的占用秒数（与 `HexwarEnv` 的常量一致）
_OCCUPY = {"attack": None, "area_fire": None, "capture": None, "hide": None,
           "load": None, "unload": None, "grenade": None, "launch": None}


class RealtimeEpisode:
    """1 Hz 实时的双席对局：每秒、双方各有一次观察 / 下令 / 改令机会。

    BLUE 先、RED 后（同一秒内），两人都决策完这一秒才 `tick_second()`。
    `reset(seed=...)` / `step(action, side=...)` 返回可 JSON 序列化的状态字典。
    """

    def __init__(self, snapshot, *, max_seconds=1200, move_radius=2, fog=True,
                 public_score=True, scoring_mode="verified", dt=1,
                 deploy_window="opening"):
        _positive_int("max_seconds", max_seconds, True)
        _positive_int("move_radius", move_radius, True)
        if not isinstance(fog, bool) or not isinstance(public_score, bool):
            raise ValueError("fog / public_score must be booleans")
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        _positive_int("dt", dt)
        if deploy_window not in DEPLOY_WINDOWS:
            raise ValueError("deploy_window must be one of %s" % (DEPLOY_WINDOWS,))
        self._deploy_window = deploy_window
        self._snapshot = snapshot
        self._max_seconds = max_seconds
        self._move_radius = move_radius
        self._fog = fog
        self._public_score = public_score
        self._scoring_mode = scoring_mode
        self._dt = dt
        self._session = None
        self._acted = 0
        self._ticks = 0
        self._outcome = None
        self._final_scores = None
        self.opening_scores = None
        self._legal_key = None   # 合法动作缓存键（tick, acted, side）——同一 tick 内 step 复用它
        self._legal = []
        self._obs_key = None     # 观测缓存键（同一 tick 内 observe 只建一次；LOS 很贵）
        self._obs = None

    # ---------------- 生命周期 ----------------
    def reset(self, *, seed=0):
        _positive_int("seed", seed, True)
        self._session = _SerialSession(self._snapshot, seed, self._max_seconds,
                                       self._fog, self._dt, "score", self._scoring_mode,
                                       self._deploy_window)
        self._acted = 0
        self._ticks = 0
        self._outcome = None
        self._final_scores = None
        self._orders = {}
        self._legal_key, self._legal = None, []
        self._obs_key, self._obs = None, None
        self.opening_scores = self._scores_now()
        self._check_outcome()
        return self._status()

    @property
    def current_actor(self):
        self._require_reset()
        if self._outcome:
            return None
        return SIDES[self._acted % 2]

    def _require_reset(self):
        if self._session is None:
            raise RuntimeError("reset must be called first")

    @staticmethod
    def _validate_side(side):
        if side not in SIDES:
            raise ValueError("side must be BLUE or RED")

    def _scores_now(self):
        score = self._session.select("BLUE").score()
        return {"BLUE": reward_value(score["own"], self._scoring_mode),
                "RED": reward_value(score["enemy"], self._scoring_mode)}

    # ---------------- 观测 ----------------
    def observe(self, side):
        self._require_reset()
        self._validate_side(side)
        key = (self._ticks, self._acted, side)
        if self._obs_key != key:
            self._obs_key, self._obs = key, self._build_observation(side)
        return _json_copy(self._obs)

    def _build_observation(self, side):
        env = self._session.select(side)
        g = env.game
        observation = _json_copy(env.observe())
        score = observation.pop("score")
        if self._public_score:
            observation["score"] = {"own": reward_value(score["own"], self._scoring_mode),
                                    "enemy": reward_value(score["enemy"], self._scoring_mode),
                                    "diff": score["diff"]}
        # ★ 每个单位补上"正在执行的指令 / 执行进度 / 剩余时间"（正式客户端每帧下发的就是这些）
        for view in observation["own"]:
            u = g.units.get(view["id"])
            view["order"] = self._order_state(u, g.second)
        public_enemy_fields = {"id", "name", "typeid", "color", "pos",
                               "hexid", "blood", "suppressed"}
        observation["enemy_visible"] = [
            {k: v for k, v in unit.items() if k in public_enemy_fields}
            for unit in observation["enemy_visible"]]
        observation["time_left"] = max(0, self._max_seconds - g.second)
        observation["tick"] = self._ticks
        observation["done"] = self._outcome is not None
        observation["current_actor"] = self.current_actor
        return observation

    @staticmethod
    def _order_state(u, now):
        """单位当前在执行什么、还剩几秒（正式规则里玩家每秒都能看到这些）。"""
        if u is None:
            return None
        state = {}
        if u.detail.get("path"):
            state["kind"] = "move"
            state["remaining_hexes"] = max(0, len(u.detail["path"]) - u.detail.get("path_idx", 0))
        if u.detail.get("target"):
            state["target"] = u.detail["target"]
            state.setdefault("kind", "engage")
        if u.detail.get("carrier"):
            state.setdefault("kind", "aboard")
        busy = float(u.detail.get("busy_until", 0)) - now
        if busy > 0:
            state["busy_seconds"] = math.ceil(busy)
            state.setdefault("kind", "busy")
        deadline = float(u.detail.get("move_until", 0)) - now
        if deadline > 0:
            state["leg_seconds"] = math.ceil(deadline)
        return state or None

    # ---------------- 合法动作 ----------------
    def legal_actions(self, side=None):
        self._require_reset()
        if side is None:
            side = self.current_actor
            if side is None:
                return []
        self._validate_side(side)
        if side != self.current_actor:
            return []                      # 同一秒里只有轮到的那一方能决策
        return _json_copy(self._legal_for(side))

    def _legal_for(self, side):
        """当前 tick 该方的合法动作（带缓存）。

        缓存是必要的：`step` 必须校验"这个动作在本席合法集合里"，
        而枚举一次约 11 ms（真实地图、半径 2、310 个候选）。同一 tick 内
        只要没人下过令，集合就不会变，所以 `(tick, acted, side)` 就是安全键。
        """
        key = (self._ticks, self._acted, side)
        if self._legal_key != key:
            self._legal_key, self._legal = key, self._enumerate(side)
        return self._legal

    def _enumerate(self, side):
        authoritative = self._session.select(side)
        view = self.observe(side)
        # 与旧接口同一套做法：只把"本方单位 + 公开可见的敌方"投影进枚举，
        # 不让枚举过程利用隐藏信息（迷雾公平性）。
        # ★ 这里做**浅拷贝**而不是深拷贝：枚举本身是只读的（不掷骰、不改状态），
        #   深拷贝整个环境在「1 Hz × 每秒 2 次枚举」下是主要开销（实测 0.093 s/次，
        #   其中 0.072 s 花在 deepcopy 上）。浅拷贝共享 `hexes` / `cities` / `rng`，
        #   只替换 `game.units` —— 置换写在副本上，不影响权威状态。
        second = authoritative.game.second
        projected = copy.copy(authoritative)
        projected_game = copy.copy(authoritative.game)
        units = own_units_for_projection(authoritative.game, side)
        # ★ 可见敌人直接用**权威对象本身**（与己方同源），不再用裁剪过的公共字段重建 Unit。
        #   旧写法（`Unit({id,name,typeid,color,pos,hexid,blood,suppressed})`）丢掉了
        #   `hide`（掩蔽）/`ste`/`e2`（滞空高度）等**判定字段**，于是投影里的
        #   `observe_range` 不减半、`unit_height` 也错，枚举会 advertise 出执行时被
        #   `can_observe` 拒绝的攻击 —— 实测随机/采样一撞就报「射击失败：不可观察」。
        #   迷雾仍然保留：只放**本 tick 可见**的敌人，看不见的照旧缺席。
        for public in view["enemy_visible"]:
            u = authoritative.game.units.get(public["id"])
            if u is not None and u.alive():
                units[u.id] = u
        projected_game.units = units
        projected.game = projected_game
        raw = projected.legal_actions(move_radius=self._move_radius,
                                      deep_validate=False)
        now = second
        out = [{"type": "keep"}]
        for act in raw:
            kind = act["type"]
            if kind == "wait":
                continue                   # 实时语义下"什么都不做"就是 keep，不需要占时间
            u = authoritative.game.units.get(act.get("unit"))
            if u is None or u.color != side:
                continue
            busy = float(u.detail.get("busy_until", 0)) > now
            if kind == "move":
                act = dict(act)
                act.pop("seconds", None)   # 持续指令：不按"整条路径耗时"计，改按秒推进
                out.append(act)
            elif kind == "launch":
                # 发射要等**弹**的部署动作做完（占用记在弹上）：上车那一秒不能同秒发射。
                m = authoritative.game.units.get(act.get("missile"))
                if m is not None and float(m.detail.get("busy_until", 0)) <= now:
                    out.append(act)
            elif kind == "stop":
                pass                       # 见下面的显式生成（`raw` 里的 stop 是回合制语义的）
            elif not busy:
                out.append(act)
        # `stop`（**停止行动**）与 `keep`（**保持现有指令**）是两件事：
        # 只要该单位还在执行指令（机动路径 / 本格未走完 / 持续打击），就允许停止。
        # 注意不能依赖 `raw` 里的 stop —— 那是回合制语义（只在 `pos2 != pos` 时给），
        # 而实时语义下状态存在 `move_until`（本段机动的到达时刻）里。
        for u in sorted(authoritative.game.units.values(), key=lambda x: str(x.id)):
            if u.color != side or u.detail.get("carrier"):
                continue
            if (u.detail.get("path") or u.detail.get("target")
                    or float(u.detail.get("move_until", 0)) > now):
                out.append({"type": "stop", "unit": u.id})
        return out

    # ---------------- 执行 ----------------
    def step(self, action, *, side=None):
        self._require_reset()
        if self._outcome is not None:
            raise RuntimeError("episode is complete; call reset to start another")
        actor = self.current_actor
        if side is not None:
            self._validate_side(side)
            if side != actor:
                raise ValueError("action is from the wrong seat")
        if not isinstance(action, dict):
            raise ValueError("action must be a JSON object")
        serialized = json.dumps(action, sort_keys=True, allow_nan=False)
        choices = [json.dumps(a, sort_keys=True, allow_nan=False)
                   for a in self._legal_for(actor)]
        if serialized not in choices:
            raise ValueError("action is not in this seat's legal actions")
        self._apply(actor, json.loads(serialized))
        self._legal_key = None             # 状态已变，缓存作废
        self._obs_key = None
        self._acted += 1
        if self._acted % 2 == 0:
            # ★ 本秒双方都决策完毕 → 时钟推进 1 秒（所有指令并行执行、自动裁决按秒结算）
            self._session.env.game.tick_second()
            self._session.refresh()
            self._ticks += 1
            self._check_outcome()
        return self._status()

    def _apply(self, side, action):
        env = self._session.select(side)
        g = env.game
        kind = action["type"]
        if kind == "keep":
            return
        u = g.units.get(action["unit"])
        if u is None:
            raise ValueError("unit does not exist")
        if kind == "move":
            path = list(action.get("path") or [])
            if path and path[0] == u.pos:
                path = path[1:]
            if not path:
                raise ValueError("move order needs at least one hex")
            ok, why = g.begin_move(u.id, path)
            if not ok:
                raise ValueError(why or "move order rejected")
            return
        if kind == "stop":
            u.detail.pop("path", None)
            u.detail.pop("path_idx", None)
            u.detail.pop("move_until", None)
            u.detail.pop("target", None)
            u.detail.pop("target_weapon", None)
            u.pos2 = u.pos
            return
        # 其余动作复用 `HexwarEnv._apply`（保证规则只有一份），用它的耗时作为**单位占用**
        cost = None
        if kind == "attack" and action.get("track"):
            _, _, _, info = env._apply(action)
            if info.get("seconds", 0) <= 0:
                raise ValueError(info.get("msg") or "order rejected")
            cost = info["seconds"]
        else:
            _, _, _, info = env._apply(action)
            if info.get("seconds", 0) <= 0 and kind not in ("stop", "stop_attack"):
                raise ValueError(info.get("msg") or "order rejected")
            cost = max(0, info.get("seconds", 0))
        if cost:
            u.detail["busy_until"] = g.second + cost
            if kind == "launch":
                # 发射要同时占住**载车**与**弹**：载车这一秒不能再下令
                # （它要完成发射动作），弹也要等发射完成才能开始机动/攻击。
                m = g.units.get(action.get("missile"))
                if m is not None:
                    m.detail["busy_until"] = g.second + cost

    # ---------------- 终局与状态 ----------------
    def _check_outcome(self):
        outcome = self._session.outcome()
        if outcome and outcome["reason"] == "time_limit":
            outcome = {"terminated": True, "truncated": False,
                       "reason": "game_clock", "winner": None}
        if outcome and outcome["terminated"]:
            self._final_scores = self._scores_now()
            if outcome["reason"] == "game_clock":
                blue, red = (self._final_scores[s] for s in SIDES)
                outcome["winner"] = None if blue == red else ("BLUE" if blue > red else "RED")
        self._outcome = outcome

    def _status(self):
        outcome = self._outcome or {"terminated": False, "truncated": False,
                                    "reason": None, "winner": None}
        return _json_copy({"current_actor": self.current_actor,
                           "tick": self._ticks,
                           "seconds": self._session.env.game.second,
                           "scoring_mode": self._scoring_mode,
                           "scoring_version": _scoring.SCORING_VERSION,
                           **outcome,
                           "final_scores": self._final_scores,
                           "opening_scores": self.opening_scores})

    def status(self):
        self._require_reset()
        return self._status()
