"""All tunable constants live here so experiments only touch one file."""
import math

WIDTH, HEIGHT = 640, 480
FPS = 60

PADDLE_WIDTH = 100
PADDLE_HEIGHT = 12
PADDLE_Y = HEIGHT - 40
PADDLE_SPEED = 8

BALL_RADIUS = 7
BALL_SPEED = 5.0                      # constant speed, pixels per step
MAX_BOUNCE_ANGLE = math.radians(60)   # angle from vertical at the paddle edge
SERVE_ANGLE = math.radians(30)

BRICK_ROWS, BRICK_COLS = 4, 8
BRICK_MARGIN, BRICK_GAP = 20, 4
BRICK_HEIGHT, BRICK_TOP = 18, 50

LIVES = 3
POINTS_PER_BRICK = 10
MAX_STEPS = 6000      # safety cap so an episode can't run forever

# Rewards: tune these experimentally later
REWARD_PADDLE_HIT = 1.0
REWARD_BRICK = 5.0
REWARD_LEVEL_COMPLETE = 30.0
REWARD_LIFE_LOST = -10.0
REWARD_GAME_OVER = -20.0

LEFT, STAY, RIGHT = 0, 1, 2   # the AI's three actions

# --- Learning hyperparameters ---
ALPHA = 0.1             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998   # multiplied in after every episode

# --- Files ---
QTABLE_FILE = "qtable.json"
STATS_FILE = "stats.json"