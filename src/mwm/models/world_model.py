import torch
from torch import nn
from mwm.models.fusion import MaskedMeanFusion
from mwm.models.encoders import MLP


class WorldModel(nn.Module):
	def __init__(self, sensor_dims, action_dim, latent_dim, hidden_dim, belief_dim):
		super().__init__()
		self.encoders = nn.ModuleDict(
				{sensor: MLP(in_dim=dim, hidden_dim=hidden_dim, out_dim=latent_dim) for sensor, dim in sensor_dims.items()}
			)
		self.fusion = MaskedMeanFusion()
		self.gru = nn.GRU(input_size=latent_dim+action_dim, hidden_size=belief_dim, batch_first=True)
		self.decoders = nn.ModuleDict(
				[[sensor, MLP(in_dim=belief_dim, hidden_dim=hidden_dim, out_dim=dim)] for sensor, dim in sensor_dims.items()]
			)
	def forward(self, obs, masks, actions):
		encoded = {}
		for sensor,x in obs.items():
			encoded[sensor] = self.encoders[sensor](x)
		fused = self.fusion(encoded, masks)
		model_input = torch.cat((fused, actions), dim=-1)
		beliefs, _ = self.gru(model_input)
		decoded = {
			sensor: decoder(beliefs) for sensor, decoder in self.decoders.items()
			}
		return decoded, beliefs