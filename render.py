"""Pygame drawing. Separate from game.py so training never needs a window."""
import pygame

import config as C
from stats import scale_points

BG = (18, 18, 28)
PANEL_BG = (26, 26, 40)
TEXT = (200, 200, 210)
ACCENT = (160, 200, 255)
GRID = (60, 60, 80)
BRICK_COLOURS = [(231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113)]


class Renderer:
    def __init__(self, panel_width=0):
        pygame.init()
        self.panel_width = panel_width
        self.screen = pygame.display.set_mode((C.WIDTH + panel_width, C.HEIGHT))
        pygame.display.set_caption("Brick Breaker RL")
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------ game area
    def _draw_game(self, game, info_lines=()):
        s = self.screen
        pygame.draw.rect(s, BG, (0, 0, C.WIDTH, C.HEIGHT))
        for (x, y, w, h) in game.bricks:
            row = int((y - C.BRICK_TOP) // (C.BRICK_HEIGHT + C.BRICK_GAP))
            pygame.draw.rect(s, BRICK_COLOURS[row % 4], (x, y, w, h), border_radius=3)
        pygame.draw.rect(s, (220, 220, 230),
                         (game.paddle_x, C.PADDLE_Y, C.PADDLE_WIDTH, C.PADDLE_HEIGHT),
                         border_radius=4)
        pygame.draw.circle(s, (255, 255, 255),
                           (int(game.ball_x), int(game.ball_y)), C.BALL_RADIUS)
        header = f"Score {game.score}   Bricks {len(game.bricks)}   Lives {game.lives}"
        s.blit(self.font.render(header, True, TEXT), (10, 8))
        for n, line in enumerate(info_lines):
            s.blit(self.font.render(line, True, ACCENT), (10, 28 + 18 * n))
        if game.done:
            msg = "LEVEL COMPLETE" if game.won else "GAME OVER"
            surf = self.font.render(msg, True, (255, 255, 255))
            s.blit(surf, surf.get_rect(center=(C.WIDTH // 2, C.HEIGHT // 2)))

    def draw(self, game, info_lines=()):
        self._draw_game(game, info_lines)
        pygame.display.flip()

    # ------------------------------------------------------- analysis panel
    def _draw_chart(self, rect, title, raw, smooth, vmin, vmax):
        x, y, w, h = rect
        s = self.screen
        s.blit(self.small.render(title, True, TEXT), (x, y - 16))
        pygame.draw.rect(s, GRID, rect, 1)
        s.blit(self.small.render(f"{vmax:.0f}", True, GRID), (x + 3, y + 2))
        s.blit(self.small.render(f"{vmin:.0f}", True, GRID), (x + 3, y + h - 15))
        if len(raw) >= 2:
            pygame.draw.lines(s, (70, 90, 130), False, scale_points(raw, x, y, w, h, vmin, vmax), 1)
            pygame.draw.lines(s, (255, 200, 80), False, scale_points(smooth, x, y, w, h, vmin, vmax), 2)

    def _draw_panel(self, stats, epsilon, reward_so_far):
        s = self.screen
        x0 = C.WIDTH
        pygame.draw.rect(s, PANEL_BG, (x0, 0, self.panel_width, C.HEIGHT))
        lines = [
            f"Episode        {len(stats) + 1}",
            f"Epsilon        {epsilon:.3f}",
            f"Reward (now)   {reward_so_far:.0f}",
            f"Avg bricks/100 {stats.recent_mean('bricks'):.1f}",
            f"Avg score/100  {stats.recent_mean('score'):.1f}",
            f"Avg reward/100 {stats.recent_mean('reward'):.1f}",
        ]
        for n, line in enumerate(lines):
            s.blit(self.font.render(line, True, TEXT), (x0 + 14, 12 + 20 * n))

        cw = self.panel_width - 28
        bricks = stats.data["bricks"]
        reward = stats.data["reward"]
        self._draw_chart((x0 + 14, 170, cw, 120), "Bricks destroyed per episode",
                         bricks, stats.moving_average("bricks", 50),
                         0, max(10, max(bricks, default=0)))
        self._draw_chart((x0 + 14, 330, cw, 120), "Episode reward (gold = 50-ep average)",
                         reward, stats.moving_average("reward", 50),
                         min(-10, min(reward, default=0)), max(10, max(reward, default=0)))

    def draw_training(self, game, stats, epsilon, reward_so_far):
        """Draw game + panel. Returns False if the user closed the window."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        self._draw_game(game)
        self._draw_panel(stats, epsilon, reward_so_far)
        pygame.display.flip()
        return True

    def tick(self, fps=C.FPS):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()