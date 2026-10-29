import numpy as np
import json
from pathlib import Path
from mwm.rollout import rollout
from mwm.envs.arena import Arena2D
from mwm.sensors.lighthouse import LighthouseSensor
from mwm.sensors.odometer import OdometrySensor

settings ={
	'num_episodes' : 1000,
	'steps_per_episode' : 500,
	'margin' : 0.3,
	'beacon_pos' : [5.0, 5.0],
	'beacon_sigma' : [0.02, 0.02],
	'odo_sigmas' : [0.01, 0.01],
	'splits' : {"train": [0,800], "val": [800,900], "test": [900,1000]},
	'wheelbase' : 0.3,
	'scale_factor' : 1.01,
	'arena_size' : 10,
	'dt' : 0.1,
	'max_turn' : 1.5,
	'max_speed' : 1.0
}

assert settings['splits']['test'][1] == settings['num_episodes']

if __name__ == "__main__":
	lighthouse = LighthouseSensor(beacon_pos=settings['beacon_pos'], 
		sigma_dist=settings['beacon_sigma'][0], sigma_bearing=settings['beacon_sigma'][1])
	odometer = OdometrySensor(sigma_left=settings['odo_sigmas'][0], sigma_right=settings['odo_sigmas'][1], 
		scale_factor=settings['scale_factor'], wheelbase=settings['wheelbase'])
	states = np.zeros([settings['num_episodes'], settings['steps_per_episode']+1, 3], dtype=np.float32)
	actions = np.zeros([settings['num_episodes'], settings['steps_per_episode'], 2], dtype=np.float32)
	observations_lighthouse = np.zeros([settings['num_episodes'], settings['steps_per_episode'], lighthouse.obs_dim], dtype=np.float32)
	observations_odometer = np.zeros([settings['num_episodes'], settings['steps_per_episode'], odometer.obs_dim], dtype=np.float32)
	for i in range(0,settings['num_episodes']):
		states[i,:,:], observations_dict, actions[i, :, :] = rollout(N_steps=settings['steps_per_episode'], 
			arena=Arena2D(size=settings['arena_size'], dt=settings['dt'], max_speed=settings['max_speed'], max_turn=settings['max_turn']), 
			seed=i, sensors=[lighthouse, odometer], margin=settings['margin'])
		observations_odometer[i,:,:] = observations_dict[odometer.name]
		observations_lighthouse[i,:,:] = observations_dict[lighthouse.name]
	Path("data").mkdir(exist_ok=True)
	np.savez_compressed(
		'data/dataset_v1.npz',
		states=states,
		actions=actions,
		obs_lighthouse=observations_lighthouse,
		obs_odometer=observations_odometer,
		)
	with open("data/dataset_v1.json", "w") as f:
		json.dump(settings, f, indent=2)
	