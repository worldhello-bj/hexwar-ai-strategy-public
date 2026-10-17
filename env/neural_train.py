"""终端回报的神经学习，按**真实比赛口径**训练。

口径只有一种（被证伪的旧做法已删，不再保留可选分支）：

* **对手 = 当前策略自己**（自博弈）。均匀随机对手一旦被超越就不再提供梯度。
* **奖励 = 净胜分增量** = (本方终−本方开) − (对方终−对方开)：敌方夺控会扣分。
* **采样流 = 每局独立播种**：局与局无关 ⇒ 可并行、可复现。
* **训练 = 成对**：同一种子一蓝一红，回报取两局净分之和（= 平台判胜负的总分差）。

Only own-seat observations enter the learner.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from engine import scoring
from engine.game import DEPLOY_WINDOWS
from .episode_api import TwoPlayerEpisode
from .match_eval import SIDES, _positive_int, snapshot_manifest
from .neural_policy import NeuralPolicy, Decision
from .public_observation import PublicObservationAdapter
from .realtime_api import RealtimeEpisode

#: 对局接口，决定了"一局里有多少个决策点"这个根本语义：
#:
#: * `serial`  = 回合制 `TwoPlayerEpisode`：一次 `step` 是一方一条指令，**时钟按这条指令
#:   自己的时长整段跳** ⇒ 一条 60 秒的机动期间对手没有任何决策点；1200 秒一局每方只有
#:   ~12–15 次决策。保留它只为历史基线可比。
#: * `realtime` = 1 Hz `RealtimeEpisode`：**每秒双方各一次**决策，机动登记为路径并在统一
#:   时间轴上按秒推进（某单位机动**不**阻塞其它单位与对手）⇒ 1200 秒一局每方 1200 次
#:   机会。与平台同粒度，是“一方移动不必等另一方”的那个语义。
INTERFACES = ("serial", "realtime")
#: `RealtimeEpisode` 接受的构造参数（`episode_config` 中多余的键丢弃，如 `max_steps`）
_REALTIME_KEYS = ("max_seconds", "move_radius", "fog", "public_score", "scoring_mode",
                  "dt", "deploy_window")


def realtime_episode_config(episode_config):
    return {k: v for k, v in episode_config.items() if k in _REALTIME_KEYS}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8], "big")


def summary(records):
    completed = [r for r in records if r["own_score"] is not None]
    return {"games": len(records), "completed": len(completed),
            "truncated": len(records) - len(completed),
            "mean_own_score": float(np.mean([r["own_score"] for r in completed])) if completed else None,
            "mean_score_diff": float(np.mean([r["score_diff"] for r in completed])) if completed else None,
            **{label: sum(r["result"] == label for r in completed) for label in ("win", "draw", "loss")}}


#: 采样随机流：**每局按 `derived_seed(seed, 局种子, 席位, "policy_sample")` 重新播种**。
#: 以前还提供过“跨局沿用同一条 `policy.rng`”的口径，它使第 k 局的随机数取决于
#: 前面跑过多少局 ⇒ 不能并行、顺序一变结果就变。已删除。

#: 学习目标（奖励）：**净胜分增量** = (本方终−本方开) − (对方终−对方开)。
#: 与平台判胜负的“净胜分”同向，并且开局席位常数被两边各扣一次后**完全抵消**。
#: 只需接口能给出双方开局分（`opening_scores`，1 Hz 实时接口有）；
#: 拿不到时退回“本方得分增量” —— 那是接口能力限制，不是可选口径。
#:
#: 已经被删掉的旧口径 `own_gain`（只扣本方开局分）：本方分里没有任何“敌方夺控”项，
#: 对手拿下夺控点、拉开分差时奖励**毫无反应** ⇒ 策略学不到“别丢点”。

#: 对手：**用当前策略自己给对手选招**（自博弈）。
#: 已经被删掉的旧做法是均匀随机抽合法动作 —— 它太弱且不自洽：练到一定程度后
#: 继续与它对练几乎不再提供梯度，策略很快撞到天花板。

#: 进程池 worker 的公共上下文（每个进程只需初始化一次 snapshot/配置）
_WORKER_STATE = {}


def policy_sample_seed(seed, episode_seed, learner_side):
    """每局按 (seed, 局种子, 席位) 派生采样种子 ⇒ 局与局相互独立、可并行、可复现。"""
    return derived_seed(seed, episode_seed, learner_side, "policy_sample")


class _RolloutRunner:
    """跑一局所需的**全部不可变上下文**（可 pickle ⇒ 能丢进进程池）。

    `NeuralTrainer._rollout` 现在只 是它的委托；并行路径把同一个对象交给子进程，
    子进程用同一段代码算 —— 所以串行与并行逐字节相同。
    """

    __slots__ = ("snapshot", "manifest", "interface", "episode_config", "observation_schema",
                 "seed", "entropy_every", "allow_real")

    def __init__(self, snapshot, manifest, interface, episode_config, observation_schema,
                 seed, entropy_every, allow_real):
        self.snapshot = snapshot
        self.manifest = manifest
        self.interface = interface
        self.episode_config = episode_config
        self.observation_schema = observation_schema
        self.seed = seed
        self.entropy_every = entropy_every
        self.allow_real = allow_real

    def sample_seed(self, episode_seed, learner_side):
        """该局的采样种子（每局独立）。"""
        return policy_sample_seed(self.seed, episode_seed, learner_side)

    def rollout(self, policy, episode_seed, learner_side, *, training):
        # 复用 `NeuralTrainer._rollout` 的**同一份实现**：它只依赖上面这 8 个不可变
        # 属性，把 runner 实例当 `self` 传进去就是同一段代码（避免两份实现漂移）。
        return NeuralTrainer._rollout(self, policy, episode_seed, learner_side,
                                      training=training)


def _worker_init(runner):
    """进程池初始化：每个子进程只接一份不可变上下文（不重算 manifest）。"""
    _WORKER_STATE["runner"] = runner


def _worker_rollout(task):
    """在子进程里跑一局。策略从检查点状态重建 ⇒ 子进程之间不共享任何可变状态。"""
    state, episode_seed, learner_side, training = task
    policy = NeuralPolicy.from_checkpoint(copy.deepcopy(state))
    return _WORKER_STATE["runner"].rollout(policy, episode_seed, learner_side,
                                          training=training)


class NeuralTrainer:
    """Resumable on-policy batches; incomplete batches survive checkpoints.

    A train call never flushes a partial batch: splitting the same episode budget
    across calls does not alter optimization. Evaluation seeds are reserved at
    construction and cannot subsequently be used as training input.
    """
    def __init__(self, snapshot, *, seed=0, train_seed_start=0, batch_episodes=4,
                 episode_config=None, policy_config=None, heldout_seeds=(1000000, 1000001),
                 observation_schema=1, allow_real=False, scoring_mode="verified",
                 interface="serial", rollout_workers=1,
                 progress_path=None, trajectory_stride=1):
        for name, value in (("seed", seed), ("train_seed_start", train_seed_start)):
            _positive_int(name, value, True)
        _positive_int("batch_episodes", batch_episodes)
        _positive_int("rollout_workers", rollout_workers)
        if rollout_workers > 64:
            raise ValueError("rollout_workers exceeds per-call safety bound")
        self.rollout_workers = rollout_workers
        # ★ 轨迹子采样：进梯度的决策只取每 k 条中的 1 条。终极奖励是**单局一个标量**，
        #   同一局里所有决策共享它，所以抽样不会丢“局级信号”，却把
        #   update 成本与检查点体积都降到 1/k（一局轨迹原本 ≈200 MB 的 JSON）。
        _positive_int("trajectory_stride", trajectory_stride)
        if trajectory_stride > 64:
            raise ValueError("trajectory_stride exceeds safety bound")
        self.trajectory_stride = trajectory_stride
        # ★ 成对训练**恒开**（真实比赛就是「红一局 + 蓝一局算总分」）：
        #   ①第 i 局用种子 `train_seed_start + i//2`、席位交替（同种子、两边各打一局）；
        #   ②**回报用这一对的和**，而不是单局：
        #        R = net(蓝局) + net(红局) = (S蓝+S红) − (O蓝+O红)
        #      开局席位常数（234/354）成对后**恰好完全抵消**，结果就是平台判胜负的总分差。
        #   两局的动作都按同一配对回报记账（比赛胜负由总分决定 ⇒ 归因应是配对级）。
        if batch_episodes % 2:
            raise ValueError("batch_episodes must be even (训练恒为成对)")
        self._pair_buffer = {}
        # ★ 进度台账（可选）：每完成一局追加一行 JSON，供 `tools/monitor_train.py` 跟读。
        #   开不开都不影响学习结果（只写文件），也不进检查点。
        self.progress_path = None if progress_path is None else str(progress_path)
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        if interface not in INTERFACES:
            raise ValueError("interface must be one of %s" % (INTERFACES,))
        self.snapshot = str(snapshot)
        self.allow_real = bool(allow_real)
        self.manifest = snapshot_manifest(snapshot, allow_real=self.allow_real)
        self.seed, self.train_seed_start, self.batch_episodes = seed, train_seed_start, batch_episodes
        self.episode_config = {"max_seconds": 1800, "max_steps": 1000, "public_score": True}
        self.episode_config.update(copy.deepcopy(episode_config or {}))
        # 奖励口径随 episode_config 一起检查点化：它决定了哪些分项能进"终局原始分"。
        self.episode_config["scoring_mode"] = scoring_mode
        self.scoring_mode = scoring_mode
        self.interface = interface
        if self.interface == "realtime":
            RealtimeEpisode(snapshot, **realtime_episode_config(self.episode_config))
        else:
            TwoPlayerEpisode(snapshot, **self.episode_config)
        self.heldout_seeds = list(heldout_seeds)
        if not self.heldout_seeds or len(self.heldout_seeds) > 256 or len(set(self.heldout_seeds)) != len(self.heldout_seeds):
            raise ValueError("reserve 1 to 256 unique heldout seeds")
        for value in self.heldout_seeds:
            _positive_int("heldout seed", value, True)
        if isinstance(observation_schema, bool) or not isinstance(observation_schema, int) or observation_schema not in (1, 2):
            raise ValueError("observation_schema must be 1 or 2")
        if self.interface == "realtime" and observation_schema != 1:
            raise ValueError("realtime interface requires observation_schema 1")
        self.observation_schema = observation_schema
        # 熵指标的采样间隔：回合制逐条（=1，历史口径），实时接口按 8 条抽一条
        # （1200 次决策/局，多算一遍前向会让单局成本翻倍）
        self.entropy_every = 1 if self.interface == "serial" else 8
        self.policy_config = copy.deepcopy(policy_config or {})
        representation = "public_v2_split" if observation_schema == 2 else "legacy_json"
        if self.policy_config.get("representation", representation) != representation:
            raise ValueError("policy representation does not match observation schema")
        if observation_schema == 2:
            self.policy_config["representation"] = representation
        self.policy = NeuralPolicy(seed=seed, **self.policy_config)
        self.initial_policy = copy.deepcopy(self.policy.checkpoint())
        self.episodes_attempted = 0
        self.episodes_completed = 0
        self.pending = []
        self.updates = 0

    def _rollout(self, policy, episode_seed, learner_side, *, training):
        if snapshot_manifest(self.snapshot, allow_real=self.allow_real) != self.manifest:
            raise ValueError("snapshot changed")
        start = time.perf_counter()
        if self.interface == "realtime":
            # 1 Hz：每秒双方各一次决策；机动登记为路径、在统一时间轴上按秒推进。
            episode = RealtimeEpisode(self.snapshot,
                                      **realtime_episode_config(self.episode_config))
            public_episode = episode
        else:
            episode = TwoPlayerEpisode(self.snapshot, **self.episode_config)
            public_episode = (PublicObservationAdapter(episode)
                              if self.observation_schema == 2 else episode)
        status = episode.reset(seed=episode_seed)
        opening = status.get("opening_scores") or {}
        opening_own = opening.get(learner_side)
        other_side = SIDES[1 - SIDES.index(learner_side)]
        opening_other = opening.get(other_side)
        # 双方都由策略选招：各自派生一条**局部**采样流，互不影响（从不碰 `policy.rng`）。
        sample_rng = np.random.default_rng(
            policy_sample_seed(self.seed, episode_seed, learner_side))
        opponent_rng = np.random.default_rng(
            derived_seed(self.seed, episode_seed, learner_side, "opponent_sample"))
        eval_rng = np.random.default_rng(derived_seed(self.seed, episode_seed, learner_side, "evaluation"))
        trajectory, entropies, counts = [], [], []
        decisions = 0
        while not status["terminated"] and not status["truncated"]:
            side = status["current_actor"]
            candidates = (public_episode.candidate_views(side)
                          if side == learner_side and self.observation_schema == 2
                          else episode.legal_actions(side))
            if not candidates:
                status = {**status, "terminated": False, "truncated": True,
                          "reason": "no_legal_actions", "winner": None, "final_scores": None,
                          "rewards": {s: 0 for s in SIDES}}
                break
            if side == learner_side:
                observation = public_episode.observe(side)
                counts.append(len(candidates))
                decisions += 1
                if training:
                    # 熵只是**描述性指标**。实时接口下每局有 1200 次决策，每次多算一遍
                    # 前向会直接把单局成本翻倍，所以按 `entropy_every` 抽样统计；
                    # 回合制（十几条指令）仍逐条算，保持与历史报告可比。
                    if decisions % self.entropy_every == 1:
                        probs = policy.probabilities(observation, candidates)
                        entropies.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-300)))))
                    # 每局一条局部流 ⇒ 局与局无关、可并行、可复现。
                    action, decision = policy.sample(observation, candidates, rng=sample_rng)
                    trajectory.append(decision)
                else:
                    probs = policy.probabilities(observation, candidates)
                    entropies.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-300)))))
                    action = candidates[int(eval_rng.choice(len(candidates), p=probs))]
            else:
                # 对手也由**当前策略**选招（自博弈）。对手的观测只用于给对手选招，
                # **绝不**传给学习者、也不进轨迹。
                if self.observation_schema == 2:
                    action = policy.sample(public_episode.observe(side),
                                           public_episode.candidate_views(side),
                                           rng=opponent_rng)[0]
                else:
                    action = policy.sample(episode.observe(side), candidates,
                                           rng=opponent_rng)[0]
            if self.observation_schema == 2:
                # schema-2 的候选是 `{"action", "public_context"}` 包，step 只收内层的 action
                action = action["action"]
            status = episode.step(action, side=side)
            # 回合制接口的奖励合约是"只在终局给分"；实时接口的 `status` 不带 rewards
            # （分数每秒都在变，按秒的增量由训练器侧的基线/差分处理）。
            if (self.interface == "serial" and not status["terminated"]
                    and any(status["rewards"].values())):
                raise ValueError("episode violated terminal-only reward contract")
        own_score = diff = result = other_score = None
        if status["terminated"] and not status["truncated"]:
            scores = status["final_scores"]
            if not isinstance(scores, dict) or any(isinstance(scores.get(s), bool) or
                    not isinstance(scores.get(s), (int, float)) or not math.isfinite(scores[s]) for s in SIDES):
                raise ValueError("completed episode requires finite terminal scores")
            own_score = scores[learner_side]
            other_score = scores[other_side]
            diff = own_score - other_score
            result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
        # 学习目标：**净胜分增量**（见文件头）。拿不到开局分（回合制接口）时退回
        # “本方得分增量” —— 那是接口能力限制，不是可选口径。
        learning_return = own_score
        own_gain = net_gain = None
        if own_score is not None and opening_own is not None:
            own_gain = own_score - opening_own
            if other_score is not None and opening_other is not None:
                net_gain = own_gain - (other_score - opening_other)
            learning_return = net_gain if net_gain is not None else own_gain
        record = {"seed": episode_seed, "learner_side": learner_side, "status": status,
                  "interface": self.interface,
                  "own_score": own_score, "other_score": other_score,
                  "opening_own": opening_own, "opening_other": opening_other,
                  "own_gain": own_gain, "net_gain": net_gain,
                  "learning_return": learning_return,
                  "score_diff": diff, "result": result,
                  "decisions": decisions, "mean_entropy": float(np.mean(entropies)) if entropies else None,
                  "mean_candidates": float(np.mean(counts)) if counts else None,
                  "max_candidates": max(counts, default=0), "elapsed_seconds": time.perf_counter() - start}
        return record, trajectory

    def train(self, episodes=20):
        _positive_int("episodes", episodes, True)
        if episodes > 10000:
            raise ValueError("episodes exceeds per-call safety bound")
        first = self.train_seed_start + self.episodes_attempted
        if any(first <= s < first + episodes for s in self.heldout_seeds):
            raise ValueError("training overlaps reserved heldout seeds")
        records = []
        if self.rollout_workers > 1:
            # 一次只并行**同一批**内的局：批内策略是冻结的，所以与串行逐局跑
            # 得到的是同一个优化序列（更新点、顺序、种子全部一致），只是墙钟变短。
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.rollout_workers, initializer=_worker_init,
                    initargs=(self._runner_context(),)) as pool:
                while len(records) < episodes:
                    records.extend(self._train_group(
                        pool, min(self.rollout_workers,
                                  episodes - len(records),
                                  max(1, self.batch_episodes - len(self.pending)))))
            return records
        for _ in range(episodes):
            index = self.episodes_attempted
            # PPO updates are atomic. Preserve the final rollout's sampling RNG
            # too, so a failed batch can be retried or checkpointed exactly.
            ppo_rng = (copy.deepcopy(self.policy.rng.bit_generator.state)
                       if self.policy.config.get("update_mode") == "ppo_clip" else None)
            seed, side = self._slot(index)
            record, trajectory = self._rollout(self.policy, seed, side, training=True)
            records.append(self._absorb(record, trajectory, ppo_rng))
        # 不在此处 flush 半对：它要等下一次调用的对面那一局，
        # 否则“拆成两次 train()”与“一次跑完”的优化序列就不一致。
        return records

    def _slot(self, index):
        """第 `index` 局的 (种子, 席位)：同一种子打两局（先蓝后红）。"""
        return self.train_seed_start + index // 2, SIDES[index % 2]

    def _runner_context(self):
        return _RolloutRunner(self.snapshot, self.manifest, self.interface,
                              self.episode_config, self.observation_schema, self.seed,
                              self.entropy_every, self.allow_real)

    def _train_group(self, pool, size):
        """并行跑 `size` 局（同批、策略冻结），按串行顺序做账。"""
        base = self.episodes_attempted
        slots = [self._slot(base + i) for i in range(size)]
        ppo_rng = (copy.deepcopy(self.policy.rng.bit_generator.state)
                   if self.policy.config.get("update_mode") == "ppo_clip" else None)
        state = self.policy.checkpoint()
        results = list(pool.map(_worker_rollout,
                                [(state, seed, side, True)
                                 for seed, side in slots]))
        return [self._absorb(record, trajectory, ppo_rng)
                for record, trajectory in results]

    def _absorb(self, record, trajectory, ppo_rng):
        """一局的账：计数 / 入批 / （满批时）更新；串行与并行共用。"""
        self.episodes_attempted += 1
        if record["own_score"] is not None:
            self.episodes_completed += 1
            stride = self.trajectory_stride
            kept = trajectory if stride == 1 else trajectory[::stride]
            self._enqueue_pair(record, kept, ppo_rng)
        self._write_progress(record)
        return record

    def _enqueue_pair(self, record, kept, ppo_rng):
        """成对训练：同一 `seed` 的蓝/红两局到齐后，用**两局净分之和**一起入批。"""
        record["pair_return"] = None
        seed, side = record.get("seed"), record.get("learner_side")
        if seed is None or side not in ("BLUE", "RED"):
            # 测试桩/老记录可能没有这两个字段 ⇒ 当成单局处理，不静默拼错对
            self._enqueue([(record, kept, record.get("learning_return"))], ppo_rng)
            return
        slot = self._pair_buffer.setdefault(seed, {})
        slot[side] = (record, kept)
        if len(slot) < 2:
            return
        self._pair_buffer.pop(seed, None)
        blue, red = slot["BLUE"], slot["RED"]
        gains = (blue[0].get("net_gain"), red[0].get("net_gain"))
        if all(g is not None for g in gains):
            pair_return = float(sum(gains))
            blue[0]["pair_return"] = red[0]["pair_return"] = pair_return
            entries = [(blue[0], blue[1], pair_return), (red[0], red[1], pair_return)]
        else:
            # 拿不到净分（如回合制接口没有开局分）⇒ 退回各自的单局目标
            entries = [(blue[0], blue[1], blue[0].get("learning_return")),
                       (red[0], red[1], red[0].get("learning_return"))]
        try:
            self._enqueue(entries, ppo_rng)
        except BaseException:
            # 入批失败（如更新报错）⇒ 整对放回缓冲，重试时才不会丢这两局。
            self._pair_buffer[seed] = slot
            raise

    def _enqueue(self, entries, ppo_rng):
        """把（可能成组的）若干条推进 `pending`；满批时更新；失败则整组回滚。"""
        start = len(self.pending)
        attempted_before = self.episodes_attempted - len(entries)
        completed_before = self.episodes_completed - len(entries)
        for record, kept, target in entries:
            self.pending.append((kept, record["own_score"] if target is None else target))
            if len(self.pending) == self.batch_episodes:
                try:
                    record["update"] = self.policy.update_batch(self.pending)
                except (FloatingPointError, OverflowError, ValueError):
                    del self.pending[start:]
                    self.episodes_attempted = attempted_before
                    self.episodes_completed = completed_before
                    if ppo_rng is not None:
                        self.policy.rng.bit_generator.state = ppo_rng
                    raise
                self.pending.clear()
                self.updates += 1

    def _write_progress(self, record):
        """追加一行进度（失败不影响训练：进度是外挂观测，不是产物）。"""
        if not self.progress_path:
            return
        row = {"wall": time.time(), "attempted": self.episodes_attempted,
               "completed": self.episodes_completed, "updates": self.updates,
               "seed": record.get("seed"), "side": record.get("learner_side"),
               "own": record.get("own_score"), "R": record.get("learning_return"),
               "own_gain": record.get("own_gain"), "net_gain": record.get("net_gain"),
               "diff": record.get("score_diff"), "result": record.get("result"),
               "decisions": record.get("decisions"),
               "episode_seconds": record.get("elapsed_seconds"),
               "truncated": bool(record.get("own_score") is None)}
        try:
            Path(self.progress_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.progress_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError:
            self.progress_path = None      # 写不了就关掉，不拖累训练

    def evaluate(self, seeds=None):
        seeds = self.heldout_seeds if seeds is None else list(seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(s not in self.heldout_seeds for s in seeds):
            raise ValueError("evaluation must use unique reserved heldout seeds")
        before = canonical(self.checkpoint())
        reports = {}
        pool = (concurrent.futures.ProcessPoolExecutor(
            max_workers=self.rollout_workers, initializer=_worker_init,
            initargs=(self._runner_context(),)) if self.rollout_workers > 1 else None)
        try:
            for label, checkpoint in (("initial", self.initial_policy), ("trained", self.policy.checkpoint())):
                policy = NeuralPolicy.from_checkpoint(copy.deepcopy(checkpoint))
                frozen = canonical(policy.checkpoint())
                pairs = [(seed, side) for seed in seeds for side in SIDES]
                if pool is None:
                    games = [self._rollout(policy, seed, side, training=False)[0]
                             for seed, side in pairs]
                else:
                    # 评测本身与顺序无关（动作由派生的 `eval_rng` 抽，策略冻结），
                    # 所以并行结果与串行**逐字段相同**。
                    state = policy.checkpoint()
                    games = [row[0] for row in pool.map(
                        _worker_rollout, [(state, seed, side, False) for seed, side in pairs])]
                if canonical(policy.checkpoint()) != frozen:
                    raise RuntimeError("evaluation mutated frozen policy")
                reports[label] = {"summary": summary(games), "by_seat": {
                    side: summary([r for r in games if r["learner_side"] == side]) for side in SIDES}, "games": games}
        finally:
            if pool is not None:
                pool.shutdown()
        if canonical(self.checkpoint()) != before:
            raise RuntimeError("evaluation mutated trainer")
        reports["heldout_seeds"] = list(seeds)
        reports["comparison"] = "Frozen initial and trained stochastic policies versus uniform random; identical paired seeds and seats."
        reports["caveat"] = "Descriptive synthetic-game sample, not a strength claim; truncations excluded from score and W/D/L summaries."
        return reports

    @property
    def algorithm(self):
        return "neural_terminal_ppo_clip_vs_random" if self.policy.config.get("update_mode") == "ppo_clip" else "neural_terminal_reinforce_vs_random"

    def checkpoint(self):
        return copy.deepcopy({"schema_version": 1, "algorithm": self.algorithm,
            **({"observation_schema": 2} if self.observation_schema == 2 else {}),
            # 计分口径绑定：口径一变，旧检查点的"原始终局分"就不可比，
            # 混用两套奖励会污染训练，所以载入时必须逐步校验。
            "scoring_version": scoring.SCORING_VERSION,
            "scoring_mode": self.scoring_mode,
            "interface": self.interface,
            "entropy_every": self.entropy_every,
            "rollout_workers": self.rollout_workers,
            "trajectory_stride": self.trajectory_stride,
            "snapshot": self.manifest, "seed": self.seed, "train_seed_start": self.train_seed_start,
            "batch_episodes": self.batch_episodes, "episode_config": self.episode_config,
            "policy_config": self.policy_config, "heldout_seeds": self.heldout_seeds,
            "episodes_attempted": self.episodes_attempted, "episodes_completed": self.episodes_completed,
            "updates": self.updates, "policy": self.policy.checkpoint(), "initial_policy": self.initial_policy,
            # ★ 半对缓冲也必须入盘：续跑时要拿它跟下一局拼对，否则这对的回报就变了。
            "pair_buffer": [{"seed": record["seed"], "side": side,
                             "return": record.get("learning_return"),
                             "net_gain": record.get("net_gain"),
                             "pair_return": record.get("pair_return"),
                             "decisions": [d.to_dict() for d in kept]}
                            for seed in sorted(self._pair_buffer, key=str)
                            for side, (record, kept) in sorted(self._pair_buffer[seed].items())],
            "pending": [{"decisions": [d.to_dict() for d in trajectory], "return": score}
                        for trajectory, score in self.pending]})

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(canonical(self.checkpoint()) + "\n")

    @classmethod
    def load(cls, path, snapshot, *, allow_real=False, interface=None, rollout_workers=None,
             progress_path=None):
        data = json.loads(Path(path).read_text("utf-8"))
        if data.get("schema_version") != 1 or data.get("algorithm") not in ("neural_terminal_reinforce_vs_random", "neural_terminal_ppo_clip_vs_random"):
            raise ValueError("unsupported checkpoint")
        saved_interface = data.get("interface", "serial")
        if interface is not None and interface != saved_interface:
            raise ValueError("requested interface differs from checkpoint")
        # 旧检查点（没有 scoring_version）是在旧计分口径下训练的，奖励不可比：
        # 直接拒绝，而不是静默拿它继续跑。
        if data.get("scoring_version") != scoring.SCORING_VERSION:
            raise ValueError(
                "checkpoint scoring version %r does not match %r; retrain or convert"
                % (data.get("scoring_version"), scoring.SCORING_VERSION))
        trainer = cls(snapshot, observation_schema=data.get("observation_schema", 1), allow_real=allow_real,
                      scoring_mode=data.get("scoring_mode", "verified"),
                      interface=saved_interface,
                      # 并行度只是速度旋钮，允许续跑时覆盖。
                      rollout_workers=(data.get("rollout_workers", 1) if rollout_workers is None
                                       else rollout_workers),
                      trajectory_stride=data["trajectory_stride"],
                      progress_path=progress_path,
                      **{k: data[k] for k in ("seed", "train_seed_start", "batch_episodes",
                                              "episode_config", "policy_config", "heldout_seeds")})
        if trainer.scoring_mode != data.get("scoring_mode", "verified"):
            raise ValueError("checkpoint scoring mode differs")
        if data["algorithm"] != trainer.algorithm:
            raise ValueError("checkpoint algorithm and update mode differ")
        if trainer.manifest != data["snapshot"]:
            raise ValueError("checkpoint snapshot does not match")
        for key in ("episodes_attempted", "episodes_completed", "updates"):
            _positive_int(key, data[key], True)
            setattr(trainer, key, data[key])
        if trainer.episodes_completed > trainer.episodes_attempted:
            raise ValueError("invalid episode counters")
        trainer.policy = NeuralPolicy.from_checkpoint(data["policy"])
        initial = NeuralPolicy.from_checkpoint(data["initial_policy"])
        expected_config = NeuralPolicy(seed=trainer.seed, **trainer.policy_config).config
        if trainer.policy.config != expected_config or initial.config != expected_config or initial.steps != 0:
            raise ValueError("checkpoint policy configuration or initialization differs")
        epochs = trainer.policy.config.get("update_epochs", 1)
        if trainer.policy.steps > trainer.updates * epochs or trainer.policy.steps % epochs:
            raise ValueError("optimizer steps exceed completed batch updates")
        if trainer.policy.config.get("baseline_offset") == "prior_batch_mean":
            if (initial.return_count != 0 or
                    trainer.policy.return_count != trainer.updates * trainer.batch_episodes):
                raise ValueError("return baseline count differs from completed batches")
        trainer.initial_policy = copy.deepcopy(data["initial_policy"])
        trainer.pending = [([Decision.from_dict(d) for d in row["decisions"]], row["return"]) for row in data["pending"]]
        trainer._pair_buffer = {}
        for row in data["pair_buffer"]:
            trainer._pair_buffer.setdefault(row["seed"], {})[row["side"]] = (
                {"seed": row["seed"], "learner_side": row["side"],
                 "learning_return": row["return"], "net_gain": row["net_gain"],
                 "pair_return": row["pair_return"]},
                [Decision.from_dict(d) for d in row["decisions"]])
        # 已完成的局 = 已入批更新掉的 + 还在 pending 的 + 还在等对面那一局的半对
        buffered = sum(len(sides) for sides in trainer._pair_buffer.values())
        if len(trainer.pending) >= trainer.batch_episodes or (
                trainer.episodes_completed
                != trainer.updates * trainer.batch_episodes + len(trainer.pending) + buffered):
            raise ValueError("invalid batch counters")
        buckets = trainer.policy.feature_width
        if any((features[0] >= buckets).any() for history, _ in trainer.pending for decision in history
               for features in (decision.observation, *decision.candidates)):
            raise ValueError("pending feature index exceeds policy buckets")
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
               for _, score in trainer.pending):
            raise ValueError("invalid pending return")
        if any(trainer.train_seed_start <= s < trainer.train_seed_start + trainer.episodes_attempted for s in trainer.heldout_seeds):
            raise ValueError("checkpoint training overlaps heldout seeds")
        return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="想定目录（比赛初设：scenario/7527；本地完整采集：snapshot/parsed）")
    parser.add_argument("--allow-real-map", action="store_true",
                        help="显式允许使用真实想定（scenario/7527 随仓库公开；本地完整采集 snapshot/parsed 不入库）")
    parser.add_argument("--move-radius", type=int, default=None,
                        help="机动枚举半径（默认 1；真实地图建议 3~6）")
    parser.add_argument("--deep-validate", action="store_true",
                        help="用参考口径逐候选事务校验枚举（慢，仅供差分复现）")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000000, 1000001])
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deploy-window", choices=DEPLOY_WINDOWS, default="opening",
                        help="巡飞弹部署（上车）窗口：opening=对抗开始前（手册 4.10(3)，默认）；"
                             "any=未发射前随时（训练里要学到用弹就选它）")
    parser.add_argument("--interface", choices=INTERFACES, default=None,
                        help="对局接口：serial=回合制（旧口径；一次 step 一条指令、时钟整段跳）；"
                             "realtime=1 Hz（每秒双方各一次决策、机动按秒并行推进，与平台同粒度）。"
                             "不传时用检查点里记的那个（新训练默认 serial）。")
    parser.add_argument("--rollout-workers", type=int, default=None,
                        help="并行 rollout 的进程数（默认 1 = 串行；续跑时默认沿用检查点里的值）。"
                             "每局独立播种 ⇒ 局与局无关、可并行、结果可复现；"
                             "并行度不会超过 batch_episodes。")
    parser.add_argument("--progress", type=Path, default=None,
                        help="可选的进度台账（JSONL，每局一行），供 tools/monitor_train.py 跟读")
    parser.add_argument("--trajectory-stride", type=int, default=1,
                        help="进梯度的决策子采样步长（默认 1=全部）。单局奖励是一个标量，"
                             "抽稀不丢局级信号；k>1 时 update 成本与检查点体积降为 1/k")
    parser.add_argument("--buckets", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--observation-schema", type=int, choices=(1, 2), default=None,
                        help="2 opts into public map/dynamic/candidate split; resume inherits saved schema")
    parser.add_argument("--baseline-offset", choices=("none", "prior_batch_mean"), default=None,
                        help="opt-in prior completed-batch score mean; resume inherits saved mode")
    parser.add_argument("--update-mode", choices=("reinforce", "ppo_clip"), default=None)
    parser.add_argument("--scoring", dest="scoring_mode", choices=("verified", "platform"),
                        default="verified",
                        help="reward scope: verified = only confirmed components (default); "
                             "platform = use the platform-style three-component sum")
    parser.add_argument("--ppo-clip", type=float, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true",
                        help="只评测一个已存在的检查点：载入 --checkpoint，跑 --eval-seeds"
                             "（省略则用检查点里预留的 heldout 种子），写 --report；不训练、不改检查点")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.eval_only:
        # ★ 只评测：神经（PPO/REINFORCE）检查点原先没有评测入口 —— `env.policy_eval`
        #   只认 MC 表检查点，于是文档里的 `ppo → eval` 流程直接报 `unsupported checkpoint`。
        #   这里补上，行为与训练收尾时的 evaluation 完全一致（同一个 `evaluate()`、
        #   同一套 heldout 种子约定），且**不改动检查点本身**。
        if not args.checkpoint.exists():
            parser.error("--eval-only 需要已存在的 --checkpoint")
        if args.report.exists():
            parser.error("--report 已存在：换个新路径")
        try:
            trainer = NeuralTrainer.load(args.checkpoint, args.snapshot,
                                         allow_real=args.allow_real_map)
            start = time.perf_counter()
            evaluation = trainer.evaluate(list(args.eval_seeds) if args.eval_seeds else None)
            report = {"algorithm": trainer.algorithm, "snapshot": trainer.manifest,
                      "observation_schema": trainer.observation_schema,
                      "interface": trainer.interface,
                      "rollout_workers": trainer.rollout_workers,
                      "scoring_mode": trainer.scoring_mode,
                      "scoring_version": scoring.SCORING_VERSION,
                      "episode_config": trainer.episode_config,
                      "policy_config": trainer.policy_config,
                      "eval_only": str(args.checkpoint),
                      "training": [], "training_summary": summary([]),
                      "evaluation": evaluation,
                      "episodes_attempted": trainer.episodes_attempted,
                      "episodes_completed": trainer.episodes_completed,
                      "updates": trainer.updates,
                      "resources": {"elapsed_seconds": time.perf_counter() - start}}
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({"evaluation": {k: evaluation[k]["summary"]
                                         for k in ("initial", "trained")},
                          "resources": report["resources"]}, sort_keys=True))
        return 0

    if args.checkpoint.exists() or args.report.exists() or args.checkpoint.resolve() == args.report.resolve():
        parser.error("choose separate new checkpoint and report paths")
    try:
        episode_config = {"max_seconds": args.max_seconds, "max_steps": args.max_steps,
                          "deploy_window": args.deploy_window}
        if args.move_radius is not None:
            episode_config["move_radius"] = args.move_radius
        if args.deep_validate:
            episode_config["deep_validate"] = True
        trainer = NeuralTrainer.load(args.resume, args.snapshot,
                                    allow_real=args.allow_real_map,
                                    rollout_workers=args.rollout_workers,
                                    progress_path=args.progress) if args.resume else NeuralTrainer(
            args.snapshot, observation_schema=args.observation_schema or 1, seed=args.seed, train_seed_start=args.train_seed_start, batch_episodes=args.batch_episodes,
            heldout_seeds=args.eval_seeds, episode_config=episode_config,
            allow_real=args.allow_real_map, scoring_mode=args.scoring_mode,
            interface=args.interface or "serial",
            rollout_workers=(args.rollout_workers or 1),
            trajectory_stride=args.trajectory_stride,
            progress_path=args.progress,
            policy_config={"buckets": args.buckets, "hidden_dim": args.hidden_dim, "learning_rate": args.learning_rate,
                           **({"baseline_offset": args.baseline_offset} if args.baseline_offset is not None else {}),
                           **{k: getattr(args, k) for k in ("update_mode", "ppo_clip", "update_epochs") if getattr(args, k) is not None}})
        if args.observation_schema is not None and args.observation_schema != trainer.observation_schema:
            raise ValueError("requested observation schema differs from resumed checkpoint")
        if args.scoring_mode != trainer.scoring_mode:
            raise ValueError("requested scoring mode differs from resumed checkpoint")
        if args.interface is not None and args.interface != trainer.interface:
            raise ValueError("requested interface differs from resumed checkpoint")
        if args.trajectory_stride != trainer.trajectory_stride:
            raise ValueError("requested trajectory stride differs from resumed checkpoint")
        if (args.baseline_offset is not None and
                args.baseline_offset != trainer.policy.config.get("baseline_offset", "none")):
            raise ValueError("requested baseline offset differs from resumed checkpoint")
        for key in ("update_mode", "ppo_clip", "update_epochs"):
            requested = getattr(args, key)
            if requested is not None and requested != trainer.policy.config.get(key, "reinforce" if key == "update_mode" else None):
                raise ValueError("requested " + key + " differs from resumed checkpoint")
        start = time.perf_counter()
        records = trainer.train(args.episodes)
        evaluation = trainer.evaluate()
        trainer.save(args.checkpoint)
        resources = {"elapsed_seconds": time.perf_counter() - start}
        try:
            import resource
            import sys
            resources["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            resources["process_peak_rss_bytes"] = None
        report = {"algorithm": trainer.algorithm, "snapshot": trainer.manifest,
                  "observation_schema": trainer.observation_schema,
                  "interface": trainer.interface,
                  "rollout_workers": trainer.rollout_workers,
                  "scoring_mode": trainer.scoring_mode,
                  "scoring_version": scoring.SCORING_VERSION,
                  "episode_config": trainer.episode_config, "policy_config": trainer.policy_config,
                  "training": records, "training_summary": summary(records), "evaluation": evaluation,
                  "episodes_attempted": trainer.episodes_attempted, "episodes_completed": trainer.episodes_completed,
                  "updates": trainer.updates, "pending_completed_episodes": len(trainer.pending), "resources": resources}
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"training": report["training_summary"], "evaluation": {k: evaluation[k]["summary"] for k in ("initial", "trained")}, "resources": resources}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
