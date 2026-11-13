import torch
import json
from mwm.training import train_step, evaluate, setup, persistence_baseline
from datetime import datetime
from pathlib import Path

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
	loaders, model, optimiser, sensor_dims, data_settings, stats = setup(settings)
	starting_loss = evaluate(model, loaders["val"], device)
	print(starting_loss)
	baseline_losses = persistence_baseline(loaders)
	history = {"starting_loss": starting_loss, "baseline_losses": baseline_losses, "settings": settings, "epochs": []}

	run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
	Path(f"runs/{run_name}").mkdir(parents=True, exist_ok=True)

	for epoch in range(settings["n_epoch"]):
		totals={}
		n_batches = 0
		for batch in loaders["train"]:
			step_log = train_step(model=model, batch=batch, optimiser=optimiser, device=device)
			for sensor, values in step_log.items():
				totals[sensor] = totals.get(sensor, 0) + values
			n_batches +=1
		train_losses = {sensor: value/n_batches for sensor, value in totals.items()}
		val_losses = evaluate(model, loaders["val"], device)
		ratios = {sensor: val_losses[sensor]/baseline_losses[sensor] for sensor in val_losses}
		epoch_dict = {"epoch": epoch, "train": train_losses, "val": val_losses, "baseline_ratio": ratios}
		history["epochs"].append(epoch_dict)
		print(epoch, train_losses, val_losses, ratios)
		with open(f"runs/{run_name}/history.json", "w") as f:
			json.dump(history, f, indent=2)
	checkpoint = {
		"model_state": model.state_dict(),
		"sensor_dims": sensor_dims,
		"settings": settings,
		"data_settings": data_settings,
		"stats": {key: (mean.tolist(), std.tolist()) for key, (mean, std) in stats.items()}
	}
	torch.save(checkpoint, f"runs/{run_name}/checkpoint.pt")