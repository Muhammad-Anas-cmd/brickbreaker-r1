"""Brick Breaker mechanics. No pygame here, so training can run fast without a window."""
import math
import random

import config as C


class Game:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Start a fresh game. This resets the GAME only, never the AI's learning."""
        self.bricks = self._make_bricks()
        self.lives = C.LIVES
        self.score = 0
        self.bricks_destroyed = 0
        self.lives_lost = 0
        self.paddle_hits = 0
        self.steps = 0
        self.done = False
        self.won = False
        self.game_over = False
        self._serve()
        return self

    def _make_bricks(self):
        usable = C.WIDTH - 2 * C.BRICK_MARGIN - (C.BRICK_COLS - 1) * C.BRICK_GAP
        w = usable / C.BRICK_COLS
        bricks = []
        for r in range(C.BRICK_ROWS):
            for c in range(C.BRICK_COLS):
                x = C.BRICK_MARGIN + c * (w + C.BRICK_GAP)
                y = C.BRICK_TOP + r * (C.BRICK_HEIGHT + C.BRICK_GAP)
                bricks.append((x, y, w, C.BRICK_HEIGHT))
        return bricks

    def _serve(self):
        """Centre the paddle, launch the ball upwards at a random angle."""
        self.paddle_x = (C.WIDTH - C.PADDLE_WIDTH) / 2
        self.ball_x = C.WIDTH / 2
        self.ball_y = C.PADDLE_Y - C.BALL_RADIUS - 1
        angle = self.rng.uniform(-C.SERVE_ANGLE, C.SERVE_ANGLE)
        self.ball_vx = C.BALL_SPEED * math.sin(angle)
        self.ball_vy = -C.BALL_SPEED * math.cos(angle)   # negative = up
        
    def step(self, action):
        """Advance one frame. action: 0 left, 1 stay, 2 right.
        Returns (reward, done). Each event is rewarded once."""
        if self.done:
            return 0.0, True
        reward = 0.0
        self.steps += 1

        # 1. Move paddle: action-1 gives -1, 0, +1. Clamp to the screen.
        self.paddle_x += (action - 1) * C.PADDLE_SPEED
        self.paddle_x = max(0, min(C.WIDTH - C.PADDLE_WIDTH, self.paddle_x))

        # 2. Move ball
        prev_y = self.ball_y
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        r = C.BALL_RADIUS

        # 3. Walls and ceiling: flip the velocity component
        if self.ball_x - r < 0:
            self.ball_x = r
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + r > C.WIDTH:
            self.ball_x = C.WIDTH - r
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y - r < 0:
            self.ball_y = r
            self.ball_vy = abs(self.ball_vy)

        # 4. Paddle: only when moving down AND crossing the paddle's top edge
        if (self.ball_vy > 0
                and prev_y + r <= C.PADDLE_Y
                and self.ball_y + r >= C.PADDLE_Y
                and self.paddle_x - r <= self.ball_x <= self.paddle_x + C.PADDLE_WIDTH + r):
            centre = self.paddle_x + C.PADDLE_WIDTH / 2
            offset = max(-1.0, min(1.0, (self.ball_x - centre) / (C.PADDLE_WIDTH / 2)))
            angle = offset * C.MAX_BOUNCE_ANGLE
            self.ball_vx = C.BALL_SPEED * math.sin(angle)
            self.ball_vy = -C.BALL_SPEED * math.cos(angle)
            self.ball_y = C.PADDLE_Y - r
            self.paddle_hits += 1
            reward += C.REWARD_PADDLE_HIT

        # 5. Bricks: at most one per frame, so one reward per brick
        for i, (bx, by, bw, bh) in enumerate(self.bricks):
            if (self.ball_x + r > bx and self.ball_x - r < bx + bw
                    and self.ball_y + r > by and self.ball_y - r < by + bh):
                # Smaller overlap = the side we hit. Bounce on that axis.
                ox = min(self.ball_x + r - bx, bx + bw - (self.ball_x - r))
                oy = min(self.ball_y + r - by, by + bh - (self.ball_y - r))
                if ox < oy:
                    side = 1 if self.ball_x > bx + bw / 2 else -1
                    self.ball_vx = abs(self.ball_vx) * side
                else:
                    side = 1 if self.ball_y > by + bh / 2 else -1
                    self.ball_vy = abs(self.ball_vy) * side
                del self.bricks[i]
                self.score += C.POINTS_PER_BRICK
                self.bricks_destroyed += 1
                reward += C.REWARD_BRICK
                break

        # 6. Terminal states
        if not self.bricks:
            reward += C.REWARD_LEVEL_COMPLETE
            self.won = self.done = True
            return reward, True

        if self.ball_y - r > C.HEIGHT:                 # ball fell off the bottom
            self.lives -= 1
            self.lives_lost += 1
            reward += C.REWARD_LIFE_LOST
            if self.lives <= 0:
                reward += C.REWARD_GAME_OVER
                self.game_over = self.done = True
                return reward, True
            self._serve()

        if self.steps >= C.MAX_STEPS:
            self.done = True
        return reward, self.done