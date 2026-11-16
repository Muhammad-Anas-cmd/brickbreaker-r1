"""Play yourself: arrows or A/D to move, R restart, Esc quit."""
import pygame
import config as C
from game import Game
from render import Renderer


def main():
    game = Game()
    view = Renderer()
    running = True
    while running:
        for e in pygame.event.get():                    # 1. read input events
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game.reset()
        keys = pygame.key.get_pressed()
        action = C.STAY
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = C.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = C.RIGHT
        game.step(action)                               # 2. update the state
        view.draw(game, ["Arrows/A-D move, R restart, Esc quit"])  # 3. draw it
        view.tick()                                     # 60 FPS
    view.close()


if __name__ == "__main__":
    main()