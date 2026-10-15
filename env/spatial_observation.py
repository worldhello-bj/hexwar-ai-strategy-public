"""Opt-in schema 3: public geometric relations, never tactical prescriptions.

Distances are odd-r hex distances, not traversable path lengths or predictions
of successful execution. A move's destination is its intended final path cell.
No authoritative dynamic state, rewards, preferred coordinates or rankings are
used. Existing schema-2 observations and candidate context remain unchanged.
"""
from engine.geometry import distance
from .public_observation import PublicObservationAdapter


def objective_relations(action, observation):
    """Pure function of an action and its seat's already-public observation."""
    unit = observation["own"].get(str(action.get("unit")))
    if unit is None or "pos" not in unit:
        return {}
    origin = int(unit["pos"])
    destination = (int(action["path"][-1])
                   if action.get("type") == "move" and action.get("path")
                   else origin)
    result = {}
    for oid, objective in sorted(observation["objectives"].items()):
        if "pos" not in objective:
            continue
        start = distance(origin, int(objective["pos"]))
        end = distance(destination, int(objective["pos"]))
        result[oid] = {
            "origin_distance": start,
            "intended_destination_distance": end,
            "distance_change": end - start,
            "held_by": objective.get("held_by"),
        }
    return result


class SpatialObservationAdapter(PublicObservationAdapter):
    schema_version = 3

    @staticmethod
    def _candidate_context(action, observation):
        context = PublicObservationAdapter._candidate_context(action, observation)
        context["objective_relations"] = objective_relations(action, observation)
        return context
