import numpy as np
import zlib
from mwm.policies import generate_action, wall_aware_action

def rollout(N_steps, arena, seed, sensors=None, margin=0.2):
	policy_rng = np.random.default_rng([seed, zlib.crc32(b"policy")])
	state = arena.reset(seed)
	if sensors is not None:
		sensor_rng = {}
		observed_vals = {}
		for sensor in sensors:
			name_number = zlib.crc32(sensor.name.encode())
			sensor_rng[sensor.name] = np.random.default_rng([seed, name_number])
			observed_vals[sensor.name] = np.zeros([N_steps, sensor.obs_dim])
			sensor.reset(state)
	else:
		observed_vals = None
	states = np.zeros([N_steps+1, 3])
	states[0,:] = state
	actions = np.zeros([N_steps, 2])
	for i in range(0,N_steps):
		action = wall_aware_action(arena, policy_rng)
		actions[i] = action
		states[i+1] = arena.step(action);
		if sensors is not None: 
			for j, sensor in enumerate(sensors):
				observed_vals[sensor.name][i,:] = sensor.observe(arena.state, arena.last_executed, sensor_rng[sensor.name])
	return states, observed_vals, actions
