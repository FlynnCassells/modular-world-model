import numpy as np
from mwm.sensors.lighthouse import LighthouseSensor

def test_lighthouse_distance():
	lighthouse = LighthouseSensor([0,0], 0, 0)
	state = np.array([3.0, 4.0, 0.0])
	rng = np.random.default_rng(0)
	distance, _ = lighthouse.observe(state, None, rng)
	assert np.isclose(distance, 5.0)

def test_lighthouse_bearing():
	lighthouse = LighthouseSensor([0,0], 0, 0)
	state = np.array([3.0, 0.0, 0.0])
	rng = np.random.default_rng(0)
	_ , bearing = lighthouse.observe(state, None, rng)
	assert np.isclose(bearing, 0.0)
	state = np.array([0.0, 3.0, 0.0])
	_ , bearing = lighthouse.observe(state, None, rng)
	assert np.isclose(bearing, np.pi/2)

def test_lighthouse_reproducible():
	lighthouse = LighthouseSensor([0,0])
	rng1 = np.random.default_rng(0)
	rng2 = np.random.default_rng(0)
	state = np.array([3.0, 4.0, 0.0])
	assert np.allclose(lighthouse.observe(state, None, rng1), lighthouse.observe(state, None, rng2))

def test_lighthouse_noise():
	lighthouse = LighthouseSensor([0,0])
	state = np.array([3.0, 4.0, 0.0])
	rng = np.random.default_rng(0)
	distances = []
	bearings = []
	for i in range(10000):
		distance, bearing = lighthouse.observe(state, None, rng)
		distances.append(distance)
		bearings.append(bearing)
	assert np.isclose(np.mean(distances), 5.0, atol=0.001)
	assert np.isclose(np.std(distances), 0.02, atol=0.001)
	assert np.isclose(np.mean(bearings), np.arctan2(4.0, 3.0), atol=0.001)
	assert np.isclose(np.std(bearings), 0.02, atol=0.001)