"""Host integration sketch; requires compatible Hexwar simulator separately.

The host owns episode and adapter. Only detached JSON crosses into GamePolicy.
This function is called for the active seat; it advances only that one action.
"""
import copy
from infer import GamePolicy

def play_one_public_turn(game_policy, public_adapter):
    side = public_adapter.current_actor
    observation = copy.deepcopy(public_adapter.observe(side))
    candidates = copy.deepcopy(public_adapter.candidate_views(side))
    selected = game_policy.choose(observation, candidates)
    # Submit the exact advertised engine action, never the feature context.
    return public_adapter.step(selected['action'], side=side)

# Host setup (compatible source revision is specified in README):
# from env.episode_api import TwoPlayerEpisode
# from env.spatial_observation import SpatialObservationAdapter
# episode = TwoPlayerEpisode(snapshot, max_seconds=1800, max_steps=1000,
#                            public_score=True, move_radius=3, scoring_mode='verified')
# adapter = SpatialObservationAdapter(episode)
# adapter.reset(seed=1000000)
# model = GamePolicy('models/frozen960-s3-seed7527-pool.npz', seed=123)
# play_one_public_turn(model, adapter)
