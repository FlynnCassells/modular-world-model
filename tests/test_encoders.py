from mwm.models.encoders import MLP
import numpy as np
from torch import nn
import torch

def test_encoder_shape():
	encoder = MLP(in_dim=3, hidden_dim=4, out_dim=64)
	x = torch.randn(8, 50, 3)
	assert encoder(x).shape == (8,50,64)

def test_encoder_output_size():
	encoder1 = MLP(in_dim=3, hidden_dim=4, out_dim=64)
	encoder2 = MLP(in_dim=2, hidden_dim=4, out_dim=64)
	x1 = torch.randn(8, 50, 3)
	x2 = torch.randn(8, 50 ,2)
	assert encoder1(x1).shape == (8, 50, 64)
	assert encoder2(x2).shape == (8,50,64)

def test_encoder_parameter_count():
	encoder = MLP(in_dim=3, hidden_dim=4, out_dim=5)
	assert sum(p.numel() for p in encoder.parameters()) == 41