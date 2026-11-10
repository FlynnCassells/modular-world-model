import torch
from torch import nn

def to_device(batch, device):
	if isinstance(batch, dict):
		return {key: to_device(value, device) for key, value in batch.items()}
	else:
		return batch.to(device)

def prepare_inputs(batch):
		sensor_inputs = {
			sensor: values[:,:-1] for sensor, values in batch["obs"].items()
		}
		actions = batch["actions"][:,1:]
		targets = {
			sensor: values[:,1:] for sensor, values in batch["obs"].items()
		}
		masks = {
			sensor: torch.ones(values.shape[:2], device=values.device) for sensor, values in sensor_inputs.items()
		}
		return sensor_inputs, actions, targets, masks

def compute_loss(predictions, targets):
	loss = {sensor: nn.functional.mse_loss(predictions[sensor], targets[sensor]) for sensor in targets}
	total = sum(loss.values())
	loss_log = {
		sensor: x.item() for sensor, x in loss.items()
	}
	return total, loss_log

def train_step(model, batch, optimiser, device):
	batch = to_device(batch, device)
	optimiser.zero_grad()
	sensor_inputs, actions, targets, masks = prepare_inputs(batch)
	predictions,_ = model(obs=sensor_inputs, masks=masks, actions=actions)
	loss, loss_log = compute_loss(predictions, targets)
	loss.backward()
	optimiser.step()
	return loss_log


def evaluate(model, loader, device):
	model.eval()
	n_batches = 0
	totals = {}
	with torch.no_grad():
		for batch in loader:
			batch = to_device(batch, device)
			sensor_inputs, actions, targets, masks = prepare_inputs(batch)
			predictions, _ = model(obs=sensor_inputs, masks=masks, actions=actions)
			loss, loss_log = compute_loss(predictions, targets)
			for sensor, values in loss_log.items():
				totals[sensor] = totals.get(sensor, 0) + value
			n_batches += 1
		model.train()
	return {sensor: value/n_batches for sensor, value in totals.items()}
