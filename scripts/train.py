import torch
import json
from mwm.models.world_model import WorldModel
from torch.utils.data import DataLoader
from mwm.training import train_step, evaluate
from mwm.data import build_datasets
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
	datasets, stats, data_settings = build_datasets(name=settings["dataset_name"], window_len=settings["window_length"], stride=settings["stride"])
	loaders = {
		split: DataLoader(ds, batch_size=settings["batch_size"], shuffle=(split=="train"))
		for split, ds in datasets.items()
	}
	sensor_dims = {
		name: t.shape[-1] for name, t in datasets["train"].obs.items()
	}
	torch.manual_seed(settings["seed"])
	model = WorldModel(sensor_dims=sensor_dims, action_dim=2, latent_dim=settings["latent_dim"], hidden_dim=settings["hidden_dim"], belief_dim=settings["belief_dim"])
	model.to(device)
	optimiser = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
	starting_loss = evaluate(model, loaders["val"], device)
	print(starting_loss)
	history = {"starting_loss": starting_loss, "settings": settings, "epochs": []}
	
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
		epoch_dict = {"epoch": epoch, "train": train_losses, "val": val_losses}
		history["epochs"].append(epoch_dict)
		print(epoch, train_losses, val_losses)
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