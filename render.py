"""Pygame drawing. Separate from game.py so training never needs a window."""
import pygame
import config as C

BG = (18, 18, 28)
BRICK_COLOURS = [(231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113)]


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
        pygame.display.set_caption("Brick Breaker RL")
        self.font = pygame.font.SysFont("consolas", 16)
        self.clock = pygame.time.Clock()

    def draw(self, game, info_lines=()):
        s = self.screen
        s.fill(BG)
        for (x, y, w, h) in game.bricks:
            row = int((y - C.BRICK_TOP) // (C.BRICK_HEIGHT + C.BRICK_GAP))
            pygame.draw.rect(s, BRICK_COLOURS[row % 4], (x, y, w, h), border_radius=3)
        pygame.draw.rect(s, (220, 220, 230),
                         (game.paddle_x, C.PADDLE_Y, C.PADDLE_WIDTH, C.PADDLE_HEIGHT),
                         border_radius=4)
        pygame.draw.circle(s, (255, 255, 255),
                           (int(game.ball_x), int(game.ball_y)), C.BALL_RADIUS)

        header = f"Score {game.score}   Bricks {len(game.bricks)}   Lives {game.lives}"
        s.blit(self.font.render(header, True, (200, 200, 210)), (10, 8))
        for n, line in enumerate(info_lines):
            s.blit(self.font.render(line, True, (160, 200, 255)), (10, 28 + 18 * n))

        if game.done:
            msg = "LEVEL COMPLETE" if game.won else "GAME OVER"
            surf = self.font.render(msg, True, (255, 255, 255))
            s.blit(surf, surf.get_rect(center=(C.WIDTH // 2, C.HEIGHT // 2)))
        pygame.display.flip()

    def tick(self, fps=C.FPS):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()