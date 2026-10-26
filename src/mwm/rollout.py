import numpy as np
from mwm.policies import generate_action, wall_aware_action

def rollout(N_steps, arena, sensor=None, seed=None):
	policy_rng, sensor_rng = np.random.default_rng(seed).spawn(2)
	arena.reset(seed)
	xs = np.zeros(N_steps)
	ys = np.zeros(N_steps)
	thetas = np.zeros(N_steps)
	obsered_vals = np.zeros([N_steps, sensor.obs_dim]) if sensor is not None else None
	for i in range(0,N_steps):
		action = wall_aware_action(arena, policy_rng)
		xs[i], ys[i], thetas[i] = arena.step(action);
		if sensor is not None: obsered_vals[i,:] = sensor.observe(arena.state, action, sensor_rng)
	return np.stack([xs, ys, thetas], axis=1), obsered_vals
