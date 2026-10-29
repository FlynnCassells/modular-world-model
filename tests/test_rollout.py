import numpy as np
from mwm.rollout import rollout
from mwm.envs.arena import Arena2D
from mwm.sensors.lighthouse import LighthouseSensor
N_steps = 500
def test_rollout():
	lighthouse = LighthouseSensor(beacon_pos=[0.0,5.0])
	states_with, _, _ = rollout(N_steps, Arena2D(), sensors=[lighthouse], seed=0)
	states_without, _, _ = rollout(N_steps, Arena2D(), seed=0)
	assert np.allclose(states_with, states_without)
def test_rollout_shapes():
	lighthouse = LighthouseSensor(beacon_pos=[0.0,5.0])
	states, observations, _ = rollout(N_steps, Arena2D(), seed=0, sensors=[lighthouse])
	assert states.shape == (N_steps+1, 3)
	assert observations[lighthouse.name].shape == (N_steps, lighthouse.obs_dim)
def test_rollout_no_sensor():
	_ , observations, _ = rollout(N_steps, Arena2D(), seed = 0)
	assert observations == None