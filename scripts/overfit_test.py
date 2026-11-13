import torch
import json
from mwm.training import train_step, evaluate, setup

settings ={
	'batch_size': 64,
	'learning_rate': 1e-3,
	'n_epoch': 10,
	"latent_dim": 64,
	"hidden_dim": 64,
	"belief_dim": 128,
	'window_length': 50,
	'stride': 10,
	'dataset_name': "dataset_v1",
	'device': "mps",
	'seed': 0
}


if __name__ =='__main__':
	device = settings["device"]
	loaders, model, optimiser, _, _, _ = setup(settings)
	starting_loss = evaluate(model, loaders["val"], device)
	print(starting_loss)
	history = {"starting_loss": starting_loss, "settings": settings, "epochs": []}
	totals={}
	n_batches = 0
	batch = next(iter(loaders["train"]))
	for i in range(300):
		step_log = train_step(model=model, batch=batch, optimiser=optimiser, device=device)
		for sensor, values in step_log.items():
			totals[sensor] = totals.get(sensor, 0) + values
		if i % 50==0:
			print(step_log)
	