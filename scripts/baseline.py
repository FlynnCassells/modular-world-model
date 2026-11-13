from mwm.training import persistence_baseline, setup

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

if __name__=="__main__":
	loaders,... = setup(settings)
	baseline_losses = persistence_baseline(settings)
	print(baseline_losses)