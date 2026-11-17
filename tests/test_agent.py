import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from agent import QAgent, discretise
from game import Game


def test_update_matches_hand_calculation():
    a = QAgent(alpha=0.5, gamma=0.9)
    s, s2 = (0, 0, 0, 0, 0), (1, 1, 1, 1, 1)
    a.q[s][2] = 1.0
    a.q[s2] = np.array([0.0, 4.0, 2.0])
    a.update(s, 2, reward=3.0, next_state=s2, terminal=False)
    # 1 + 0.5 * (3 + 0.9*4 - 1) = 3.8
    assert abs(a.q[s][2] - 3.8) < 1e-9


def test_terminal_update_ignores_future():
    a = QAgent(alpha=1.0, gamma=0.9)
    s, s2 = (0, 0, 0, 0, 0), (1, 1, 1, 1, 1)
    a.q[s2] = np.array([100.0, 100.0, 100.0])
    a.update(s, 0, reward=-20.0, next_state=s2, terminal=True)
    assert a.q[s][0] == -20.0


def test_greedy_picks_best_action():
    a = QAgent(epsilon=0.0, seed=1)
    s = (0, 0, 0, 0, 0)
    a.q[s] = np.array([0.0, 0.0, 5.0])
    assert all(a.choose_action(s, training=True) == 2 for _ in range(20))


def test_full_exploration_tries_all_actions():
    a = QAgent(epsilon=1.0, seed=1)
    s = (0, 0, 0, 0, 0)
    a.q[s] = np.array([0.0, 0.0, 5.0])
    assert {a.choose_action(s) for _ in range(200)} == {0, 1, 2}


def test_evaluation_never_explores():
    a = QAgent(epsilon=1.0, seed=1)
    s = (0, 0, 0, 0, 0)
    a.q[s] = np.array([0.0, 5.0, 0.0])
    assert all(a.choose_action(s, training=False) == 1 for _ in range(50))


def test_epsilon_decays_to_floor():
    a = QAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(10):
        a.decay_epsilon()
    assert a.epsilon == 0.1


def test_save_and_load_roundtrip(tmp_path):
    a = QAgent(epsilon=0.3)
    a.q[(1, 2, 0, 1, 3)] = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "q.json"
    a.save(p)
    b = QAgent()
    b.load(p)
    assert b.epsilon == 0.3
    assert np.allclose(b.q[(1, 2, 0, 1, 3)], [1.0, 2.0, 3.0])


def test_discretise_in_range_over_a_game():
    g = Game(seed=1)
    for _ in range(500):
        bx, by, vx, vy, px = discretise(g)
        assert 0 <= bx < 8 and 0 <= by < 6 and vx in (0, 1, 2) and vy in (0, 1) and 0 <= px < 8
        g.step(C.RIGHT)
        if g.done:
            break