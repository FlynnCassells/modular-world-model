import numpy as np
from mwm.envs.arena import Arena2D

def test_forward_motion_east():
	arena = Arena2D()
	arena.reset(seed=0)
	arena.state = np.array([5.0, 5.0, 0.0])
	x, y, theta = arena.step([1.0, 0.0])
	assert np.isclose(x, 5.1)
	assert np.isclose(y, 5.0)
def test_wall():
	arena = Arena2D()
	arena.reset(seed=0)
	arena.state = np.array([9.9, 5.0, 0.0])
	for i in range (0,100):
		x, y, theta = arena.step([1.0, 0.0])
		assert 0.0 <= x <= 10.0
		assert 0.0 <= y <= 10.0
def test_seed():
	arena = Arena2D()
	state1 = arena.reset(seed=1)
	state2 = arena.reset(seed=1)
	assert np.allclose(state1, state2)
def test_no_reverse():
	arena = Arena2D()
	state_i = arena.reset(seed=0);
	arena.step([-1.0, 0])
	assert np.allclose(state_i, arena.state)
def test_turn_limited():
	arena = Arena2D()
	arena.reset(seed=0)
	arena.state = np.array([5.0, 5.0, 0])
	x, y, theta = arena.step([0, 100.0])
	assert np.isclose(theta, arena.max_turn*arena.dt)