import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from game import Game


def test_reset_state():
    g = Game(seed=1)
    assert len(g.bricks) == C.BRICK_ROWS * C.BRICK_COLS
    assert g.lives == C.LIVES and g.score == 0 and not g.done


def test_ball_speed_is_constant():
    g = Game(seed=2)
    for _ in range(300):
        g.step(C.STAY)
        assert abs(math.hypot(g.ball_vx, g.ball_vy) - C.BALL_SPEED) < 1e-6
        if g.done:
            break


def test_paddle_stays_in_bounds():
    g = Game(seed=3)
    for _ in range(200):
        g.step(C.LEFT)
    assert g.paddle_x == 0
    for _ in range(200):
        g.step(C.RIGHT)
    assert g.paddle_x == C.WIDTH - C.PADDLE_WIDTH


def test_paddle_hit_rewarded_once():
    g = Game(seed=4)
    g.bricks = [g.bricks[0]]
    g.ball_x, g.ball_y = g.paddle_x + 50, C.PADDLE_Y - C.BALL_RADIUS - 2
    g.ball_vx, g.ball_vy = 0.0, C.BALL_SPEED
    total = sum(g.step(C.STAY)[0] for _ in range(5))
    assert g.paddle_hits == 1 and total == C.REWARD_PADDLE_HIT and g.ball_vy < 0


def test_brick_rewarded_once_and_removed():
    g = Game(seed=5)
    bx, by, bw, bh = g.bricks[-1]
    n = len(g.bricks)
    g.ball_x, g.ball_y = bx + bw / 2, by + bh + C.BALL_RADIUS + 1
    g.ball_vx, g.ball_vy = 0.0, -C.BALL_SPEED
    total = sum(g.step(C.STAY)[0] for _ in range(3))
    assert len(g.bricks) == n - 1 and total == C.REWARD_BRICK and g.ball_vy > 0


def test_game_over_after_three_lives():
    g = Game(seed=6)
    for _ in range(C.LIVES):
        g.ball_x, g.ball_y = 10, C.HEIGHT + C.BALL_RADIUS + 1
        g.ball_vx, g.ball_vy = 0.0, C.BALL_SPEED
        r, done = g.step(C.STAY)
    assert done and g.game_over
    assert r == C.REWARD_LIFE_LOST + C.REWARD_GAME_OVER