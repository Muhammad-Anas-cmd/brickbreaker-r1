"""Tabular Q-learning agent. Knows nothing about pygame or game internals."""
import json
import random
from collections import defaultdict

import numpy as np

import config as C

N_ACTIONS = 3
X_BINS = 8       # ball x
Y_BINS = 6       # ball y
VX_BINS = 3      # ball moving left / roughly vertical / right
PADDLE_BINS = 8  # paddle x


def discretise(game):
    """Turn the game's continuous numbers into a small tuple of integers (the state)."""
    bx = min(X_BINS - 1, int(game.ball_x / C.WIDTH * X_BINS))
    by = min(Y_BINS - 1, max(0, int(game.ball_y / C.PADDLE_Y * Y_BINS)))
    if game.ball_vx < -1.0:
        vx = 0
    elif game.ball_vx > 1.0:
        vx = 2
    else:
        vx = 1
    vy = 1 if game.ball_vy > 0 else 0          # 1 = moving towards the paddle
    px = min(PADDLE_BINS - 1, int(game.paddle_x / (C.WIDTH - C.PADDLE_WIDTH + 1) * PADDLE_BINS))
    return (bx, by, vx, vy, px)


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS))   # state -> [Q(left), Q(stay), Q(right)]

    def choose_action(self, state, training=True):
        """Epsilon-greedy while training, purely greedy when evaluating."""
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self.q[state]
        best = np.flatnonzero(values == values.max())       # break ties randomly
        return int(self.rng.choice(list(best)))

    def update(self, state, action, reward, next_state, terminal):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        future = 0.0 if terminal else self.gamma * self.q[next_state].max()
        td_error = reward + future - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        data = {
            "epsilon": self.epsilon,
            "q": {",".join(map(str, s)): v.tolist() for s, v in self.q.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.epsilon = data["epsilon"]
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS))
        for key, v in data["q"].items():
            self.q[tuple(int(k) for k in key.split(","))] = np.array(v)