"""Opt-in schema 4: shared public intended-endpoint flags, not capture predictions.

The endpoint is the last supplied move path cell, otherwise the acting unit's
public position. No acting unit/position means no endpoint. “Unowned” means an
explicit public held_by=None, not enemy-owned or an unknown ownership field.
Ground type 1/2 includes artillery and says nothing about capture eligibility,
health, transport, contest, reachability, or successful action execution.
"""
from .spatial_observation import SpatialObservationAdapter


def objective_contacts(action, observation):
    """Pure aggregate of supplied public data, with keys shared across IDs."""
    unit = observation["own"].get(str(action.get("unit")))
    endpoint = None
    if unit is not None and "pos" in unit:
        endpoint = (int(action["path"][-1])
                    if action.get("type") == "move" and action.get("path")
                    else int(unit["pos"]))
    matches = [objective for objective in observation["objectives"].values()
               if endpoint is not None and "pos" in objective
               and int(objective["pos"]) == endpoint]
    return {
        "intended_endpoint_is_public_objective": bool(matches),
        "intended_endpoint_is_unowned_public_objective": any(
            "held_by" in objective and objective["held_by"] is None
            for objective in matches),
        "acting_unit_is_ground_type1or2": (
            unit is not None and str(unit.get("typeid")) in ("1", "2")),
    }


class ContactObservationAdapter(SpatialObservationAdapter):
    schema_version = 4

    @staticmethod
    def _candidate_context(action, observation):
        context = SpatialObservationAdapter._candidate_context(action, observation)
        context["objective_contacts"] = objective_contacts(action, observation)
        return context
