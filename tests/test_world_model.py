from mwm.models.world_model import WorldModel
import torch
from torch import nn
import numpy as np


def test_output_shapes():
	sensor_dims = {
		"a": 3,
		"b": 2
	}
	obs = {
		"a": torch.randn(4, 10, 3),
		"b": torch.randn(4,10,2)
	}
	masks = {
		"a": torch.ones(4,10),
		"b": torch.ones(4,10)
	}
	actions = torch.randn(4,10,2)
	model = WorldModel(sensor_dims=sensor_dims, action_dim=2, latent_dim=6, hidden_dim=5, belief_dim=7)
	decoded, beliefs = model(obs=obs, masks=masks, actions=actions)
	assert decoded["a"].shape == (4,10,3)
	assert decoded["b"].shape == (4,10,2)
	assert beliefs.shape == (4,10,7)

def test_missing_sensor():
	sensor_dims = {
		"a": 3,
		"b": 2
	}
	obs = {
		"b": torch.randn(4,10,2)
	}
	masks = {
		"b": torch.ones(4,10)
	}
	actions = torch.randn(4,10,2)
	model = WorldModel(sensor_dims=sensor_dims, action_dim=2, latent_dim=6, hidden_dim=5, belief_dim=7)
	decoded, beliefs = model(obs=obs, masks=masks, actions=actions)
	assert set(decoded) == {"a", "b"}

def test_no_NaN():
	sensor_dims = {
		"a": 3,
		"b": 2
	}
	obs = {
		"a": torch.randn(4, 10, 3),
		"b": torch.randn(4,10,2)
	}
	masks = {
		"a": torch.zeros(4,10),
		"b": torch.zeros(4,10)
	}
	actions = torch.randn(4,10,2)
	model = WorldModel(sensor_dims=sensor_dims, action_dim=2, latent_dim=6, hidden_dim=5, belief_dim=7)
	decoded, beliefs = model(obs=obs, masks=masks, actions=actions)
	for name, pred in decoded.items():
		assert torch.isfinite(pred).all(), f"non-finite prediction for {name}"
	assert torch.isfinite(beliefs).all()

def test_parameter_grads():
	sensor_dims = {
		"a": 3,
		"b": 2
	}
	obs = {
		"a": torch.randn(4, 10, 3),
		"b": torch.randn(4,10,2)
	}
	masks = {
		"a": torch.ones(4,10),
		"b": torch.ones(4,10)
	}
	actions = torch.randn(4,10,2)
	model = WorldModel(sensor_dims=sensor_dims, action_dim=2, latent_dim=6, hidden_dim=5, belief_dim=7)
	decoded, beliefs = model(obs=obs, masks=masks, actions=actions)
	total = sum(pred.sum() for pred in decoded.values())
	total.backward()
	assert len(list(model.parameters())) > 0, "Model has no parameters"
	for name,p in model.named_parameters():
		assert p.grad is not None, f"No gradient for {name}"
		assert p.grad.abs().sum() > 0, f"Gradients all zero for {name}"








