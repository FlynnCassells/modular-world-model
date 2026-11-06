import torch
from torch import nn
from mwm.models.fusion import MaskedMeanFusion
import numpy as np

def test_all_ones():
	sensor_names = ["a", "b"]
	masks = {}
	encoded = {}
	for name in sensor_names:
		masks[name] = torch.ones(8,8)
		encoded[name] = torch.randn(8,8,64)
	expected = (encoded["a"]+encoded["b"])/2
	fusion = MaskedMeanFusion()
	result = fusion(encoded, masks)
	assert result.shape == (8,8,64)
	assert np.allclose(expected, result)

def test_all_zeros():
	sensor_names = ["a", "b"]
	masks = {}
	encoded = {}
	for name in sensor_names:
		masks[name] = torch.zeros(8,8)
		encoded[name] = torch.randn(8,8,64)
	expected = np.zeros((8,8,64))
	fusion = MaskedMeanFusion()
	result = fusion(encoded, masks)
	assert result.shape == (8,8,64)
	assert np.allclose(expected, result)

def test_zero_mask():
	sensor_names = ["a", "b"]
	masks = {}
	encoded={}
	masks["a"] = torch.ones(8,8)
	masks["b"] = torch.zeros(8,8)
	for name in sensor_names:
		encoded[name] = torch.randn(8,8,64)
	expected = encoded["a"]
	fusion = MaskedMeanFusion()
	result = fusion(encoded, masks)
	assert result.shape == (8,8,64)
	assert np.allclose(expected, result)

def test_varying_mask():
	encoded={
		"a": torch.randn(8,8,64),
		"b": torch.randn(8,8,64),
		}

	masks={
		"a": torch.cat((torch.ones(8,4), torch.zeros(8,4)), dim=1),
		"b": torch.ones(8,8)
		}
	expected_first = (encoded["a"][:,:4]+encoded["b"][:,:4])/2
	expected_second = encoded["b"][:,4:]
	expected = torch.cat((expected_first, expected_second), dim=1)
	fusion = MaskedMeanFusion()
	result = fusion(encoded, masks)
	assert np.allclose(expected, result)





