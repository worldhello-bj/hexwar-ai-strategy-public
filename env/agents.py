# -*- coding: utf-8 -*-
"""参考智能体：给策略搜索一个基线，也给 LLM 一个可模仿的动作范式。"""
import random
from engine import geometry as G


class BaseAgent:
    name = "base"

    def act(self, env, obs):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """随机合法动作。用来验证环境不会崩、并给出随机基线分数。"""
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, env, obs):
        acts = env.legal_actions()
        # 偏向机动，否则大部分时间在等待
        moves = [a for a in acts if a["type"] == "move"]
        pool = moves if (moves and self.rng.random() < 0.75) else acts
        return self.rng.choice(pool) if pool else {"type": "wait"}


class GreedyAgent(BaseAgent):
    """贪心基线：能打就打（挑最优战果），否则朝最近夺控点推进。"""

    name = "greedy"

    def __init__(self, prefer="attack"):
        self.prefer = prefer

    def act(self, env, obs):
        acts = env.legal_actions()
        if not acts:
            return {"type": "wait"}

        # 1) 夺控优先
        for a in acts:
            if a["type"] == "capture":
                return a

        # 2) 射击：优先打「射程近 + 对方血少 + 我方武器强」的目标
        atks = [a for a in acts if a["type"] == "attack"]
        if atks:
            atks.sort(key=lambda a: (a["dist"], _target_blood(obs, a["target"])))
            return atks[0]

        # 3) 机动：朝最近的未占领夺控点前进
        targets = [o["pos"] for o in obs["objectives"] if o.get("held_by") != obs["side"]]
        if not targets:
            targets = [o["pos"] for o in obs["objectives"]]
        moves = _one_step_moves(acts)
        if moves and targets:
            best = min(moves, key=lambda a: min(G.distance(a["path"][-1], t)
                                                for t in targets))
            return {"type": "move", "unit": best["unit"], "path": best["path"]}
        return acts[0]


class MultiUnitGreedy(GreedyAgent):
    """每回合给所有己方单位各下一条命令（更接近真实指挥）。"""

    name = "greedy-multi"

    def act(self, env, obs):
        acts = env.legal_actions()
        if not acts:
            return {"type": "wait"}
        by_unit = {}
        for a in acts:
            if a["type"] == "wait":
                continue
            uid = a.get("unit")
            if uid is None:
                continue
            by_unit.setdefault(uid, []).append(a)
        orders = []
        for unit, alist in by_unit.items():
            cap = [a for a in alist if a["type"] == "capture"]
            if cap:
                orders.append(cap[0])
                continue
            atk = [a for a in alist if a["type"] == "attack"]
            if atk:
                atk.sort(key=lambda a: (a["dist"], _target_blood(obs, a["target"])))
                orders.append(atk[0])
                continue
            mv = _one_step_moves(alist)
            targets = [o["pos"] for o in obs["objectives"]
                       if o.get("held_by") != obs["side"]] or \
                      [o["pos"] for o in obs["objectives"]]
            if mv and targets:
                best = min(mv, key=lambda a: min(G.distance(a["path"][-1], t)
                                                 for t in targets))
                orders.append({"type": "move", "unit": unit,
                               "path": best["path"]})
        return orders if orders else {"type": "wait"}


def _one_step_moves(acts):
    """legal_actions 给出的 path 含起点，所以「走一格」= path 长度为 2 且末位不同。"""
    out = []
    for a in acts:
        p = a.get("path") or []
        if a.get("type") == "move" and len(p) == 2 and p[0] != p[1]:
            out.append(a)
    return out


def _target_blood(obs, tid):
    for u in obs["enemy_visible"]:
        if u["id"] == tid:
            return u["blood"]
    return 99


AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "greedy-multi": MultiUnitGreedy,
}
