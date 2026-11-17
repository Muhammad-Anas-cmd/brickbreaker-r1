"""Per-episode statistics and the maths behind the live chart. No pygame here."""
import json

KEYS = ("reward", "score", "bricks", "lives_lost", "epsilon", "won")


class Stats:
    def __init__(self):
        self.data = {k: [] for k in KEYS}

    def __len__(self):
        return len(self.data["reward"])

    def add(self, **values):
        for k in KEYS:
            self.data[k].append(values[k])

    def moving_average(self, key, window):
        """Average of the last `window` values at each point (fewer at the start)."""
        out, total, vals = [], 0.0, self.data[key]
        for i, v in enumerate(vals):
            total += v
            if i >= window:
                total -= vals[i - window]
            out.append(total / min(i + 1, window))
        return out

    def recent_mean(self, key, window=100):
        vals = self.data[key][-window:]
        return sum(vals) / len(vals) if vals else 0.0

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f)

    def load(self, path):
        with open(path) as f:
            loaded = json.load(f)
        self.data = {k: list(loaded[k]) for k in KEYS}


def scale_points(values, x, y, w, h, vmin, vmax):
    """Map a list of numbers to (pixel_x, pixel_y) inside the box (x, y, w, h).
    Higher values are drawn higher up (smaller pixel y)."""
    n = len(values)
    if n == 0:
        return []
    span = (vmax - vmin) or 1.0
    pts = []
    for i, v in enumerate(values):
        px = x + (w * i / (n - 1) if n > 1 else 0)
        frac = max(0.0, min(1.0, (v - vmin) / span))
        pts.append((px, y + h - frac * h))
    return pts