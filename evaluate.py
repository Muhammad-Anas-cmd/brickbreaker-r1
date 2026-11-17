"""Compare an untrained agent with your trained one (no exploration, no learning).

  python evaluate.py              numbers only
  python evaluate.py --watch      then watch the trained agent play
"""
import argparse

import config as C
from agent import discretise
from game import Game
from train import make_agent, run_episode


def evaluate(agent, n, seed):
    game = Game(seed=seed)
    runs = [run_episode(agent, game, training=False) for _ in range(n)]
    avg = lambda k: sum(r[k] for r in runs) / n
    return {"bricks": avg("bricks"), "score": avg("score"),
            "reward": avg("reward"), "wins": sum(r["won"] for r in runs)}


def watch(agent):
    import pygame
    from render import Renderer
    game, view = Game(), Renderer()
    while True:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            break
        if game.done:
            game.reset()
        state = discretise(game)
        game.step(agent.choose_action(state, training=False))
        view.draw(game, ["Trained agent (no exploration). Close window to quit."])
        view.tick()
    view.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()

    untrained = make_agent(seed=0)
    untrained.epsilon = 0.0              # empty Q-table: every action ties, so it moves randomly
    trained = make_agent(seed=0)
    trained.load(C.QTABLE_FILE)

    print(f"Average over {args.games} games each (greedy, no learning):")
    for name, ag in (("untrained", untrained), ("trained  ", trained)):
        r = evaluate(ag, args.games, seed=123)
        print(f"  {name}  bricks {r['bricks']:5.1f}  score {r['score']:6.1f}  "
              f"reward {r['reward']:7.1f}  levels won {r['wins']}/{args.games}")
    if args.watch:
        watch(trained)


if __name__ == "__main__":
    main()