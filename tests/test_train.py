import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from game import Game
from stats import Stats, scale_points
from train import make_agent, run_episode


def test_training_episode_updates_q_table_and_resets_game():
    agent, game = make_agent(seed=1), Game(seed=1)
    run_episode(agent, game, training=True)
    n_states = len(agent.q)
    assert n_states > 0
    r = run_episode(agent, game, training=True)           # same Game object, reset inside
    assert len(agent.q) >= n_states                       # Q-table kept, not wiped
    assert set(r) == {"reward", "score", "bricks", "lives_lost", "won"}


def test_evaluation_does_not_learn():
    agent, game = make_agent(seed=1), Game(seed=1)
    agent.epsilon = 0.0
    run_episode(agent, game, training=False)
    assert len(agent.q) == 0 or all(v.sum() == 0 for v in agent.q.values())


def test_episode_always_terminates():
    agent, game = make_agent(seed=2), Game(seed=2)
    run_episode(agent, game, training=True)
    assert game.done and game.steps <= C.MAX_STEPS


def test_moving_average():
    s = Stats()
    for v in (1, 2, 3, 4, 5):
        s.add(reward=v, score=0, bricks=v, lives_lost=0, epsilon=1.0, won=0)
    assert s.moving_average("reward", 2) == [1.0, 1.5, 2.5, 3.5, 4.5]
    assert s.recent_mean("bricks", 3) == 4.0


def test_stats_save_load(tmp_path):
    s = Stats()
    s.add(reward=1.5, score=10, bricks=1, lives_lost=2, epsilon=0.9, won=0)
    s.save(tmp_path / "s.json")
    t = Stats()
    t.load(tmp_path / "s.json")
    assert t.data == s.data


def test_scale_points_maps_into_box():
    pts = scale_points([0, 5, 10], x=100, y=50, w=200, h=100, vmin=0, vmax=10)
    assert pts == [(100, 150), (200, 100), (300, 50)]      # higher value = smaller pixel y