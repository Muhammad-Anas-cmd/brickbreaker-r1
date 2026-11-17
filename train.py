"""Train the Q-learning agent.

  python train.py                       fast, headless
  python train.py --episodes 5000
  python train.py --live                watch it learn, with the analysis panel
  python train.py --resume              continue from saved qtable.json / stats.json

Press Ctrl+C (or close the window in --live mode) to stop safely: progress is saved.
"""
import argparse
import os

import config as C
from agent import QAgent, discretise
from game import Game
from stats import Stats


def run_episode(agent, game, training, on_frame=None):
    """Play one full game. Resets the GAME, never the agent's Q-table.
    on_frame(game, reward_so_far) is called each step; return False to stop early."""
    game.reset()
    state = discretise(game)
    total = 0.0
    while not game.done:
        action = agent.choose_action(state, training)
        reward, _ = game.step(action)
        next_state = discretise(game)
        if training:
            # Terminal = the game truly ended. Losing a life is NOT terminal.
            agent.update(state, action, reward, next_state, game.won or game.game_over)
        state = next_state
        total += reward
        if on_frame is not None and on_frame(game, total) is False:
            raise KeyboardInterrupt
    return {
        "reward": total,
        "score": game.score,
        "bricks": game.bricks_destroyed,
        "lives_lost": game.lives_lost,
        "won": int(game.won),
    }


def make_agent(seed=None):
    return QAgent(alpha=C.ALPHA, gamma=C.GAMMA, epsilon=C.EPSILON_START,
                  epsilon_min=C.EPSILON_MIN, epsilon_decay=C.EPSILON_DECAY, seed=seed)


def save_progress(agent, stats):
    agent.save(C.QTABLE_FILE)
    stats.save(C.STATS_FILE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--live", action="store_true", help="show the game and analysis panel")
    p.add_argument("--skip", type=int, default=8, help="in --live mode, draw every Nth frame")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    agent, stats, game = make_agent(args.seed), Stats(), Game(seed=args.seed)
    if args.resume and os.path.exists(C.QTABLE_FILE):
        agent.load(C.QTABLE_FILE)
        if os.path.exists(C.STATS_FILE):
            stats.load(C.STATS_FILE)
        print(f"Resumed: {len(stats)} episodes done, epsilon={agent.epsilon:.3f}")

    view, on_frame, frame = None, None, [0]
    if args.live:
        from render import Renderer          # imported here so headless runs need no window
        view = Renderer(panel_width=340)

        def on_frame(g, reward_so_far):
            frame[0] += 1
            if frame[0] % args.skip:
                return True
            return view.draw_training(g, stats, agent.epsilon, reward_so_far)

    start = len(stats)
    try:
        for ep in range(start, start + args.episodes):
            result = run_episode(agent, game, training=True, on_frame=on_frame)
            stats.add(epsilon=agent.epsilon, **result)
            agent.decay_epsilon()
            n = ep + 1
            if n % args.save_every == 0:
                save_progress(agent, stats)
                print(f"episode {n:5d}  eps {agent.epsilon:.3f}  "
                      f"avg bricks(100) {stats.recent_mean('bricks'):5.1f}  "
                      f"avg reward(100) {stats.recent_mean('reward'):7.1f}  "
                      f"states {len(agent.q)}")
    except KeyboardInterrupt:
        print("\nStopped. Saving progress...")
    save_progress(agent, stats)
    print(f"Saved {len(stats)} episodes to {C.QTABLE_FILE} and {C.STATS_FILE}")
    if view:
        view.close()


if __name__ == "__main__":
    main()