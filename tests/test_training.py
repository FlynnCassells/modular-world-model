import torch
import numpy as np
from torch import nn
from mwm.training import compute_loss, prepare_inputs, to_device
import pytest

def numbered(shape):
	size = np.prod(shape)
	return np.arange(size, dtype=np.float32).reshape(shape)

def test_compute_loss():
	predictions_equal = {"a": torch.ones(3, 8), "b": torch.ones(3,8)}
	targets_equal = predictions_equal
	equal_loss, _ = compute_loss(predictions_equal, targets_equal)
	assert np.isclose(equal_loss, 0)
	predictions_unequal = {"a": torch.zeros(3,8), "b": torch.zeros(3,8)}
	targets_unequal = {"a": 2*torch.ones(3,8), "b": torch.ones(3,8)}
	unequal_loss, unequal_log = compute_loss(predictions_unequal, targets_unequal)
	assert np.isclose(unequal_loss.item(), 5.0)
	assert np.isclose(unequal_log["a"], 4.0)
	assert np.isclose(unequal_log["b"], 1.0)

def test_prepare_inputs_alignment():
	T = 10
	obs = {"a": torch.from_numpy(numbered((2,T,3)))}
	batch = {
		"obs": obs,
		"actions": torch.from_numpy(numbered((2,T,2))),
	}
	sensor_inputs, actions, targets, masks = prepare_inputs(batch)
	assert torch.equal(obs["a"][:,4], targets["a"][:,3])
	assert torch.equal(batch["actions"][:,4], actions[:,3])
	assert torch.equal(sensor_inputs["a"][:,3], obs["a"][:,3])
	assert masks["a"].shape == (2,9)
	assert targets["a"].shape == (2,9,3)
	assert sensor_inputs["a"].shape == (2,9,3)

def test_to_device():
	dictionary = {
		"obs" : {"a": torch.ones(2,10,3)},
		"actions" : torch.ones(2,10,2)
	}
	moved_dict = to_device(dictionary, "cpu")
	assert list(moved_dict.keys()) == list(dictionary.keys())
	assert list(moved_dict["obs"].keys()) == list(dictionary["obs"].keys())
	assert torch.equal(moved_dict["obs"]["a"], dictionary["obs"]["a"])
	assert torch.equal(moved_dict["actions"], dictionary["actions"])

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_to_device_mps():
	dictionary = {
		"obs" : {"a": torch.ones(2,10,3)},
		"actions" : torch.ones(2,10,2)
	}
	moved_dict = to_device(dictionary, "mps")
	assert moved_dict["obs"]["a"].device.type == "mps"
	assert moved_dict["actions"].device.type == "mps"