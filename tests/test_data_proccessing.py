import numpy as np
from mwm.data import *

def test_lighthouse_proccess():
	rng = np.random.default_rng(0)
	arrays = {
		"states": rng.normal(10.,3.0, size=(10,50,3)),
		"obs_a": rng.normal(5.0, 2.0, size=(10,50,3)),
		"obs_b": rng.normal(0.0,1.0, size=(10,50,2)),
		"obs_lighthouse": rng.normal(3.0, 1.0, size=(10,50,2))
	}
	new = preprocess(arrays)
	lh = new["obs_lighthouse"]
	x = lh[...,1]**2 + lh[...,2]**2
	assert np.allclose(x, 1)

def test_stats_shape():
	rng = np.random.default_rng(0)
	arrays = {
		"obs_a": rng.normal(5.0, 2.0, size=(10,50,3)),
		"obs_b": rng.normal(0.0,1.0, size=(10,50,2)),
	}
	keys = ["obs_a", "obs_b"]
	stats = compute_norm_stats(arrays, keys=keys)
	assert stats["obs_a"][0].shape == (3,)
	assert stats["obs_a"][1].shape == (3,) 
	assert stats["obs_b"][0].shape == (2,) 
	assert stats["obs_b"][1].shape == (2,) 

def test_norm_stats():
	rng = np.random.default_rng(0)
	arrays = {
		"obs_a": rng.normal(5.0, 2.0, size=(10,50,3)),
		"obs_b": rng.normal(0.0,1.0, size=(10,50,2)),
	}
	keys = ["obs_a", "obs_b"]
	stats = compute_norm_stats(arrays, keys=keys)
	norms = normalise(arrays, stats)
	norm_stats = compute_norm_stats(norms, keys)
	mean_a, std_a = norm_stats["obs_a"]
	mean_b, std_b = norm_stats["obs_b"]
	assert np.allclose(mean_a, 0)
	assert np.allclose(mean_b, 0)
	assert np.allclose(std_a, 1)
	assert np.allclose(std_b, 1)

def test_states_unchanged():
	rng = np.random.default_rng(0)
	arrays = {
		"states": rng.normal(10.,3.0, size=(10,50,3)),
		"obs_a": rng.normal(5.0, 2.0, size=(10,50,3)),
		"obs_b": rng.normal(0.0,1.0, size=(10,50,2)),
		"obs_lighthouse": rng.normal(3.0, 1.0, size=(10,50,2))
	}
	original_states = arrays["states"].copy()
	keys = ["obs_a", "obs_b"]
	stats = compute_norm_stats(arrays, keys=keys)
	norms = normalise(arrays, stats)
	new = preprocess(arrays)
	assert np.array_equal(original_states,arrays["states"])