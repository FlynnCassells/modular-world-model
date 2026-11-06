import numpy as np
from mwm.data import preprocess, compute_norm_stats, normalise, WindowDataset

def generate_fake_arrays(n_episodes=10, n_steps=50):
	rng = np.random.default_rng(0)
	arrays = {
		"states": numbered([n_episodes,n_steps+1,3]),
		"actions": numbered([n_episodes,n_steps,2]),
		"obs_a": numbered([n_episodes,n_steps,3]),
		"obs_b": numbered([n_episodes,n_steps,2]),
		"obs_lighthouse": numbered([n_episodes,n_steps,2])
	}
	return arrays
def numbered(shape):
	size = np.prod(shape)
	return np.arange(size, dtype=np.float32).reshape(shape)

def test_lighthouse_proccess():
	arrays = generate_fake_arrays()
	new = preprocess(arrays)
	lh = new["obs_lighthouse"]
	x = lh[...,1]**2 + lh[...,2]**2
	assert np.allclose(x, 1)
	assert new["obs_lighthouse"].shape == (10,50,3)

def test_stats_shape():
	rng = np.random.default_rng(0)
	arrays = generate_fake_arrays()
	keys = ["obs_a", "obs_b"]
	stats = compute_norm_stats(arrays, keys=keys)
	assert stats["obs_a"][0].shape == (3,)
	assert stats["obs_a"][1].shape == (3,) 
	assert stats["obs_b"][0].shape == (2,) 
	assert stats["obs_b"][1].shape == (2,) 

def test_norm_stats():
	arrays = generate_fake_arrays()
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
	arrays = generate_fake_arrays()
	original_states = arrays["states"].copy()
	keys = ["obs_a", "obs_b"]
	stats = compute_norm_stats(arrays, keys=keys)
	norms = normalise(arrays, stats)
	new = preprocess(arrays)
	assert np.array_equal(norms["states"],new["states"])


def test_window_number():
	arrays = generate_fake_arrays()
	ds = WindowDataset(arrays, window_len=5, stride=5)
	assert len(ds) == 100

def test_window_shapes():
	arrays = generate_fake_arrays()
	ds = WindowDataset(arrays, window_len=5, stride=5)
	assert ds[0]["states"].shape == (6,3)
	assert ds[0]["actions"].shape == (5,2)
	assert ds[0]["obs"]["a"].shape == (5,3)

def test_uneven_windows():
	arrays = generate_fake_arrays(n_episodes=2,n_steps=12)
	ds = WindowDataset(arrays, window_len=5, stride=2)
	assert ds.index == [(0,0),(0,2),(0,4),(0,6),(1,0),(1,2),(1,4),(1,6)]


def test_window_sensor_names():
	arrays = generate_fake_arrays()
	ds = WindowDataset(arrays, window_len=5, stride=5)
	assert set(ds[0]["obs"]) == {"a", "b", "lighthouse"}

def test_window_content():
	arrays = generate_fake_arrays()
	ds = WindowDataset(arrays, window_len=5, stride=5)
	assert np.array_equal(ds[0]["obs"]["a"],arrays["obs_a"][0,0:5])
	assert np.array_equal(ds[13]["states"],arrays["states"][1,15:21])
	assert np.array_equal(ds[13]["actions"],arrays["actions"][1,15:20])

