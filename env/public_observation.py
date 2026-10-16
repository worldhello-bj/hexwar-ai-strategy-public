"""Opt-in schema-v2 public inputs; not wired into existing trained policies.

Wrap an existing ``TwoPlayerEpisode`` with ``PublicObservationAdapter(episode)``.
The full static board is public under this offline simulation's contract, not a
claim that every online game exposes it. Only elevation/cond/step are exported
from the map: arbitrary snapshot fields and dynamic occupancy are never copied.
``step_code`` is a static category, NOT a claim a particular unit can enter now.

Entity collections are keyed by stable string IDs. Packed positions and category
codes are strings; ``position`` supplies numeric row/col using engine geometry.
Paths remain ordered. Candidates contain ``action`` (the untouched engine action)
and detached ``public_context``. Submit candidate["action"], never the context or
whole candidate. This Python wrapper is an API boundary, not a security sandbox.
"""
import copy

from engine.geometry import split_mapid
from .episode_api import TwoPlayerEpisode


_OWN_FIELDS = frozenset({
    "id", "army", "name", "typeid", "pos", "hexid", "blood", "suppressed",
    "color", "score", "armor", "hidden", "ste", "loaded_into", "carrying",
    "engaging", "weapons", "can_observe_range",
    # 手册 4.10 巡飞弹生命周期（显式暴露，不静默丢失）：状态、在飞占用、剩余巡飞秒
    "missile_state", "missile_in_flight", "loiter_seconds_left",
})
_VISIBLE_FIELDS = frozenset({
    "id", "name", "typeid", "color", "pos", "hexid", "blood", "suppressed",
})
_KNOWN_FIELDS = frozenset({"id", "pos", "hexid", "blood", "last_seen", "stale"})
_OBJECTIVE_FIELDS = frozenset({"id", "name", "pos", "hexid", "held_by"})


def _position(mapid):
    row, col = split_mapid(mapid)
    return {"id": str(int(mapid)), "row": row, "col": col}


def _entities(items, fields):
    result = {}
    for item in items:
        entity = {key: copy.deepcopy(value) for key, value in item.items()
                  if key in fields}
        entity["id"] = str(entity["id"])
        if "pos" in entity:
            entity["position"] = _position(entity["pos"])
            entity["pos"] = entity["position"]["id"]
        for key in ("typeid", "ste", "loaded_into", "engaging"):
            if key in entity and entity[key] is not None:
                entity[key] = str(entity[key])
        if "carrying" in entity:
            entity["carrying"] = sorted(str(uid) for uid in entity["carrying"])
        if "weapons" in entity:
            weapons = {}
            for weapon in entity["weapons"]:
                view = {key: value for key, value in weapon.items()
                        if key in {"id", "name", "ammo_type", "range", "ammo"}}
                view["id"] = str(view["id"])
                if "ammo_type" in view:
                    view["ammo_type"] = str(view["ammo_type"])
                weapons[view["id"]] = view
            entity["weapons"] = dict(sorted(weapons.items()))
        result[entity["id"]] = entity
    return dict(sorted(result.items()))


class PublicObservationAdapter:
    """Detached richer observations with unchanged episode actions and payouts."""

    schema_version = 2

    def __init__(self, episode):
        if not isinstance(episode, TwoPlayerEpisode):
            raise TypeError("episode must be a TwoPlayerEpisode")
        self._episode = episode

    @property
    def current_actor(self):
        return self._episode.current_actor

    def reset(self, *, seed=0):
        return self._episode.reset(seed=seed)

    def status(self):
        return self._episode.status()

    def step(self, action, *, side=None):
        """Delegate the exact supplied action, without enrichment or rewriting."""
        return self._episode.step(action, side=side)

    def legal_actions(self, side=None):
        return self._episode.legal_actions(side)

    def _board(self):
        # Access only the loaded static board; never serialize game/unit state.
        cells = {}
        for mapid, cell in sorted(self._episode._session.env.game.hexes.items()):
            position = _position(mapid)
            cells[position["id"]] = {
                **position,
                "elevation": int(cell.get("elevation") or 0),
                "terrain_code": str(int(cell.get("cond") or 0)),
                "step_code": str(int(cell.get("step") or 0)),
            }
        rows = [cell["row"] for cell in cells.values()]
        cols = [cell["col"] for cell in cells.values()]
        return {"coordinate_system": "odd-r", "cells": cells,
                "extents": ({"min_row": min(rows), "max_row": max(rows),
                             "min_col": min(cols), "max_col": max(cols)}
                            if cells else None)}

    def observe(self, side):
        observation = self._episode.observe(side)  # validates lifecycle and seat
        result = {key: copy.deepcopy(observation[key])
                  for key in ("time", "time_left", "side", "done", "score")
                  if key in observation}
        result.update({"schema_version": self.schema_version, "board": self._board()})
        for name, fields in (("own", _OWN_FIELDS), ("enemy_visible", _VISIBLE_FIELDS),
                             ("enemy_known", _KNOWN_FIELDS),
                             ("objectives", _OBJECTIVE_FIELDS)):
            result[name] = _entities(observation[name], fields)
        return result

    @staticmethod
    def _candidate_context(action, observation):
        cells = observation["board"]["cells"]

        def cell_at(mapid):
            position = _position(mapid)
            return copy.deepcopy(cells.get(position["id"], position))

        # Positions come exclusively from this seat's public observation. In
        # particular, never resolve an action target against authoritative units.
        entities = {**observation["enemy_known"], **observation["enemy_visible"],
                    **observation["own"]}
        context = {}
        for role in ("unit", "target", "carrier"):
            entity = entities.get(str(action.get(role)))
            if role in action and entity is not None:
                context[role] = {"id": entity["id"],
                                 "position": cell_at(entity["pos"])}
                if entity.get("stale"):
                    context[role].update(stale=True, last_seen=entity["last_seen"])
        if "units" in action:
            context["units"] = {
                str(uid): {"id": str(uid), "position": cell_at(entities[str(uid)]["pos"])}
                for uid in sorted(action["units"], key=str) if str(uid) in entities}
        if "path" in action:
            context["path"] = [cell_at(mapid) for mapid in action["path"]]
        if "target_hex" in action:
            context["target_hex"] = cell_at(action["target_hex"])
        return context

    def candidate_views(self, side=None):
        """Return original actions paired with detached public spatial context."""
        actions = self._episode.legal_actions(side)
        if not actions:
            return []
        observation = self.observe(self.current_actor if side is None else side)
        return [{"action": action,
                 "public_context": self._candidate_context(action, observation)}
                for action in actions]
