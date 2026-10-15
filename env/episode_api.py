"""Detached, serial two-seat episode interface with terminal-only scoring.

This is an API isolation contract for cooperative callers, not a Python sandbox.
The match clock is part of the game definition: reaching it completes a scored
match. ``max_steps`` is an external safety budget and cannot award a result.
Existing ``match_eval`` evaluation/truncation conventions are unchanged.
"""
import copy
import json

from engine.game import Unit
from engine import scoring as _scoring
from .hexwar_env import (ATTACK_SEC, CAPTURE_SEC, HIDE_SEC, LOAD_SEC,
                         GRENADE_SEC, AREA_FIRE_SEC)

from .match_eval import SIDES, _SerialSession, _json_copy, _positive_int


reward_value = _scoring.reward_value


class TwoPlayerEpisode:
    """BLUE-first alternating decisions over one shared clock, with fog enabled.

    ``reset(seed=...)`` and ``step(action, side=...)`` return detached status
    dictionaries. Fetch a seat's input with ``observe(side)`` and its choices
    with ``legal_actions(side)``. Inactive seats have no currently legal actions.
    Intermediate rewards are zero; a natural endpoint or game clock completion
    pays each side its unmodified engine total once. Truncation pays nothing and
    has no final scores. ``public_score=True`` exposes totals-only scoreboard
    observations as an explicit game rule; False omits them. Neither option
    changes the terminal-only reward. No online scoreboard UI is assumed verified.

    Choices are enumerated on a visible-state projection. Advertised intents
    blocked by concealed state consume their normal action duration (advertised
    seconds for moves, action constants otherwise, dt as fallback), capped by
    the remaining clock, without revealing rejection details. This wrapper-only
    rule prevents cost-free hidden-state probes; it does not change the engine.
    Invalid/unadvertised actions raise ValueError without consuming a step or changing state.
    Calling step before reset or after completion raises RuntimeError. A reset
    is explicit and starts a new, independent history and seeded random stream.
    """

    def __init__(self, snapshot, *, max_seconds=120, max_steps=100,
                 dt=10, move_radius=1, victory="score", public_score=True,
                 deep_validate=False, scoring_mode="verified"):
        _positive_int("max_seconds", max_seconds, True)
        _positive_int("max_steps", max_steps, True)
        _positive_int("dt", dt)
        _positive_int("move_radius", move_radius, True)
        if victory not in ("score", "capture"):
            raise ValueError("victory must be score or capture")
        if not isinstance(public_score, bool):
            raise ValueError("public_score must be a boolean")
        if not isinstance(deep_validate, bool):
            raise ValueError("deep_validate must be a boolean")
        if scoring_mode not in ("verified", "platform"):
            raise ValueError("scoring_mode must be 'verified' or 'platform'")
        self._scoring_mode = scoring_mode
        self._public_score = public_score
        self._deep_validate = deep_validate
        self._snapshot = snapshot
        self._max_seconds = max_seconds
        self._max_steps = max_steps
        self._dt = dt
        self._move_radius = move_radius
        self._victory = victory
        self._session = None
        self._steps = 0
        self._outcome = None
        self._final_scores = None

    def _require_reset(self):
        if self._session is None:
            raise RuntimeError("reset must be called first")

    @staticmethod
    def _validate_side(side):
        if side not in SIDES:
            raise ValueError("side must be BLUE or RED")

    @property
    def current_actor(self):
        self._require_reset()
        return None if self._outcome else SIDES[self._steps % 2]

    def reset(self, *, seed=0):
        _positive_int("seed", seed, True)
        session = _SerialSession(self._snapshot, seed, self._max_seconds,
                                 True, self._dt, self._victory, self._scoring_mode)
        self._session = session
        self._steps = 0
        self._outcome = None
        self._final_scores = None
        self._check_outcome()
        return self._status(payout=True)

    def observe(self, side):
        """Return just this seat's fog-filtered view, with no live engine objects."""
        self._require_reset()
        self._validate_side(side)
        observation = _json_copy(self._session.select(side).observe())
        # Public totals are an explicit configurable game rule, not a claim
        # about a particular online UI. Detailed private score components stay out.
        score = observation.pop("score")
        if self._public_score:
            # 公开比分与奖励用**同一个口径**：默认 `verified`（只用已实测分项），
            # 以免未实测值既进观察又进奖励。口径本身是元数据，只在 `status()`
            # 里给出（`scoring_mode`），不塞进公开比分，保持观察结构不变。
            observation["score"] = {
                "own": reward_value(score["own"], self._scoring_mode),
                "enemy": reward_value(score["enemy"], self._scoring_mode),
                "diff": score["diff"]}
        public_enemy_fields = {"id", "name", "typeid", "color", "pos",
                               "hexid", "blood", "suppressed"}
        observation["enemy_visible"] = [
            {key: value for key, value in unit.items() if key in public_enemy_fields}
            for unit in observation["enemy_visible"]]
        observation["done"] = self._outcome is not None
        return observation

    def legal_actions(self, side=None):
        """Return detached enumerated actions for the actor, or [] for other seats."""
        self._require_reset()
        if side is None:
            side = self.current_actor
            if side is None:
                return []
        self._validate_side(side)
        if side != self.current_actor:
            return []
        authoritative = self._session.select(side)
        view = self.observe(side)
        # Share the immutable static map instead of deep-copying it. On the real
        # scenario that is 2721 cells per call, and this projection is built on
        # every enumeration; the same rule is already load-bearing in
        # `HexwarEnv._try_actions`, which never copies the map either.
        projected = copy.deepcopy(authoritative,
                                  {id(authoritative.game.hexes): authoritative.game.hexes})
        # Build opposing tokens from the public whitelist, not copied private
        # ammunition, carrier state, equipment, targets, or hidden occupancy.
        units = {uid: unit for uid, unit in projected.game.units.items()
                 if unit.color == side}
        for public in view["enemy_visible"]:
            unit = Unit(public)
            unit.stunned_until = (projected.game.second + 1
                                  if public["suppressed"] else 0)
            units[unit.id] = unit
        projected.game.units = units
        return _json_copy(projected.legal_actions(move_radius=self._move_radius,
                                                 deep_validate=self._deep_validate))

    def status(self):
        """Read metadata/final scores without paying a terminal reward again."""
        self._require_reset()
        return self._status(payout=False)

    def _check_outcome(self):
        outcome = self._session.outcome()
        if outcome and outcome["reason"] == "time_limit":
            outcome = {"terminated": True, "truncated": False,
                       "reason": "game_clock", "winner": None}
        if outcome and outcome["terminated"]:
            score = self._session.select("BLUE").score()
            self._final_scores = {
                "BLUE": reward_value(score["own"], self._scoring_mode),
                "RED": reward_value(score["enemy"], self._scoring_mode)}
            if outcome["reason"] == "game_clock":
                blue, red = (self._final_scores[s] for s in SIDES)
                outcome["winner"] = None if blue == red else ("BLUE" if blue > red else "RED")
        if outcome is None and self._steps >= self._max_steps:
            outcome = {"terminated": False, "truncated": True,
                       "reason": "step_budget", "winner": None}
        self._outcome = outcome

    def _status(self, *, payout):
        outcome = self._outcome or {"terminated": False, "truncated": False,
                                    "reason": None, "winner": None}
        return _json_copy({"current_actor": self.current_actor,
                           "steps": self._steps,
                           "seconds": self._session.env.game.second,
                           "scoring_mode": self._scoring_mode,
                           "scoring_version": _scoring.SCORING_VERSION,
                           **outcome,
                           "final_scores": self._final_scores,
                           "rewards": self._final_scores if payout and self._final_scores is not None
                           else {side: 0 for side in SIDES}})

    def step(self, action, *, side=None):
        """Accept exactly one advertised JSON action from the current actor.

        JSON canonical comparison intentionally distinguishes bools from integers
        and forbids extra keys or altered display metadata in advertised actions.
        """
        self._require_reset()
        if self._outcome is not None:
            raise RuntimeError("episode is complete; call reset to start another")
        actor = self.current_actor
        if side is not None:
            self._validate_side(side)
            if side != actor:
                raise ValueError("action is from the wrong seat")
        try:
            if not isinstance(action, dict):
                raise ValueError("action must be a JSON object")
            serialized = json.dumps(action, sort_keys=True, allow_nan=False)
            choices = self.legal_actions(actor)
            if serialized not in [json.dumps(a, sort_keys=True, allow_nan=False) for a in choices]:
                raise ValueError("action is not in this seat's legal actions")
            detached = json.loads(serialized)
        except (TypeError, OverflowError) as exc:
            raise ValueError("action must be a legal JSON object") from exc
        env = self._session.select(actor)
        _, _, _, info = env.step(detached)
        if info.get("invalid_action"):
            # An advertised intent may fail against concealed occupancy. The
            # atomic engine rejection is followed by a normal-duration clock
            # advance, preventing free hidden-state probes. No reason is exposed.
            costs = {"attack": ATTACK_SEC, "capture": CAPTURE_SEC,
                     "hide": HIDE_SEC, "load": LOAD_SEC, "unload": LOAD_SEC,
                     "grenade": GRENADE_SEC, "area_fire": AREA_FIRE_SEC}
            seconds = detached.get("seconds", costs.get(detached["type"], self._dt))
            env.step({"type": "wait", "seconds": max(1, seconds)})
        self._steps += 1
        self._session.refresh()
        self._check_outcome()
        return self._status(payout=True)
