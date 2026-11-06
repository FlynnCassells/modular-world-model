import torch
from torch import nn


class MaskedMeanFusion(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, encoded, masks):
		sensor_names = list(encoded.keys())
		total_activated = torch.clamp(torch.stack([masks[name] for name in sensor_names]).sum(dim=0), min=1).unsqueeze(-1)
		masked_vectors = [torch.mul(encoded[name], masks[name].unsqueeze(-1)) for name in sensor_names]
		masked_sum = torch.stack(masked_vectors).sum(dim=0)
		return masked_sum/total_activated
