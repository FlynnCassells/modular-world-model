import numpy as np
from mwm.envs.arena import Arena2D
from mwm.sensors.lighthouse import LighthouseSensor

def generate_action(arena, rng):
    v = rng.uniform(0, arena.max_speed)
    omega = rng.uniform(-arena.max_turn, arena.max_turn)
    return np.array([v, omega])


def wall_aware_action(arena, rng, margin=0.3):
    x, y, theta = arena.state
    if margin < x < arena.size - margin and margin < y < arena.size - margin:
        return generate_action(arena, rng)
    dx = arena.size / 2 - x
    dy = arena.size / 2 - y
    center_heading = np.arctan2(dy, dx)
    error = np.arctan2(np.sin(center_heading - theta), np.cos(center_heading - theta))
    omega = np.sign(error) * arena.max_turn
    v, _ = generate_action(arena, rng)
    return np.array([v, omega])
