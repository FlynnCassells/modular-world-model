import numpy as np
from mwm.envs.arena import Arena2D
from mwm.sensors.odometer import OdometrySensor


def step_and_observe(arena, state, action, scale_factor):
	odometer = OdometrySensor(scale_factor=scale_factor)
	arena.reset()
	sensor_rng = np.random.default_rng(0)
	arena.state = state
	final_state = arena.step(action)
	dL, dR = odometer.observe(state, action, sensor_rng)*arena.dt
	return final_state, np.array([dL, dR])

def test_straight_equal_wheels():
	arena = Arena2D()
	initial_state = np.array([arena.size/2, arena.size/2, 0.0])
	action = np.array([1.0, 0.0])
	scale_factor = 1.0
	initial_state = np.array([arena.size/2, arena.size/2, 0.0])
	final_state, odo_obs = step_and_observe(arena=arena, state=initial_state, action=action, scale_factor=scale_factor)
	dL, dR = odo_obs
	assert np.isclose(dL, action[0]*arena.dt)
	assert np.isclose(dR, action[0]*arena.dt)

def test_spin_on_spot():
	arena = Arena2D()
	initial_state = np.array([arena.size/2, arena.size/2, 0.0])
	action = np.array([0.0, 1.0])
	scale_factor = 1.0
	final_state, odo_obs = step_and_observe(arena=arena, state=initial_state, action=action, scale_factor=scale_factor)
	dL, dR = odo_obs
	assert np.isclose(dL, -dR)
	assert np.isclose(dL, -action[1]*arena.dt*0.3/2)

def test_wheel_mismatch():
	arena = Arena2D()
	initial_state = np.array([arena.size/2, arena.size/2, 0.0])
	action = np.array([1.0, 0.0])
	scale_factor = 1.01
	final_state, odo_obs = step_and_observe(arena=arena, state=initial_state, action=action, scale_factor=scale_factor)
	dL, dR = odo_obs
	assert np.isclose(1.01*dL, dR)

def test_wall_slip():
	arena = Arena2D()
	initial_state = np.array([arena.size, arena.size/2, 0.0])
	action = np.array([1.0, 0.0])
	scale_factor = 1.0
	final_state, odo_obs = step_and_observe(arena=arena, state=initial_state, action=action, scale_factor=scale_factor)
	dL, dR = odo_obs
	assert np.allclose(np.array([arena.size, arena.size/2, 0.0]), final_state)
	assert np.isclose(dL, 0.1)
	assert np.isclose(dR, 0.1)



