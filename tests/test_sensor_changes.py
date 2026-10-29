import numpy as np
from mwm.envs.arena import Arena2D
from mwm.rollout import rollout
from mwm.sensors.lighthouse import LighthouseSensor
from mwm.sensors.odometer import OdometrySensor

N_steps=500

def test_odometer_with_and_without_multiple_sensors():
	lighthouse = LighthouseSensor(beacon_pos=[0.0,5.0])
	odometer = OdometrySensor(scale_factor=1.0, sigma_left=0.01, sigma_right=0.01)
	seed = 0
	_, obs_with_both, _ = rollout(N_steps, Arena2D(), seed=0, sensors=[lighthouse, odometer])
	_, obs_just_odo, _ = rollout(N_steps, Arena2D(), seed=0, sensors=[odometer])
	assert np.allclose(obs_with_both[odometer.name], obs_just_odo[odometer.name])