"""Passive counters of selected actions and already-supplied public observations.

No episode/engine reference, observation fetch, policy call, RNG, or mutation.
Requested paths are not executed travel. Ownership comparisons span consecutive
learner decision observations, including any intervening opponent actions.
"""
from collections import Counter
from engine.geometry import distance


def _entities(collection):
    return collection.values() if isinstance(collection, dict) else collection


class EpisodeBehaviorDiagnostics:
    def __init__(self):
        self.action_counts = Counter()
        self.move_units = set()
        self.path_cells = self.path_segments = 0
        self.contact_observations = self.contact_present = 0
        self.objective_comparisons = self.objective_changes = 0
        self.previous_owners = None
        self.previous_positions = None
        self.position_comparisons = self.position_changes = self.hex_displacement = 0
        self.position_changed_units = set()

    def observe_decision(self, observation, action):
        # Only these explicitly public fields are consulted. Never consult stale
        # enemy_known contacts, candidate context, or private/authoritative state.
        self.action_counts[action['type']] += 1
        if action['type'] == 'move':
            self.move_units.add(str(action['unit']))
            path = action.get('path') or []
            self.path_cells += len(path)
            self.path_segments += max(0, len(path) - 1)
        if 'enemy_visible' in observation:
            self.contact_observations += 1
            self.contact_present += bool(observation['enemy_visible'])
        if 'own' in observation:
            positions = {str(item['id']): int(item['pos'])
                         for item in _entities(observation['own']) if item.get('pos') is not None}
            if self.previous_positions is not None:
                shared = positions.keys() & self.previous_positions.keys()
                self.position_comparisons += len(shared)
                for key in shared:
                    displacement = distance(positions[key], self.previous_positions[key])
                    self.hex_displacement += displacement
                    if displacement:
                        self.position_changes += 1
                        self.position_changed_units.add(key)
            self.previous_positions = positions
        else:
            self.previous_positions = None
        if 'objectives' in observation:
            owners = {str(item['id']): item['held_by']
                      for item in _entities(observation['objectives']) if 'held_by' in item}
            if self.previous_owners is not None:
                shared = owners.keys() & self.previous_owners.keys()
                self.objective_comparisons += len(shared)
                self.objective_changes += sum(owners[key] != self.previous_owners[key] for key in shared)
            self.previous_owners = owners
        else:
            self.previous_owners = None

    def report(self):
        return {
            'schema_version': 1,
            'learner_action_type_counts': dict(sorted(self.action_counts.items())),
            'unique_move_requested_units': len(self.move_units),
            'requested_move_path_cell_count': self.path_cells,
            'requested_move_path_consecutive_segment_count': self.path_segments,
            'visible_contact_observation_count': self.contact_observations,
            'visible_contact_present_decisions': self.contact_present,
            'own_position_comparisons_between_learner_observations': self.position_comparisons,
            'own_position_changes_between_learner_observations': self.position_changes,
            'unique_own_units_with_observed_position_change': len(self.position_changed_units),
            'own_hex_displacement_between_learner_observations_sum': self.hex_displacement,
            'public_objective_owner_comparisons': self.objective_comparisons,
            'public_objective_owner_changes_between_learner_observations': self.objective_changes,
        }
