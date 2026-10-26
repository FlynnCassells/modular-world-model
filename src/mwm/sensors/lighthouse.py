import numpy as np
from mwm.sensors.base import Sensor

class LighthouseSensor(Sensor):
	def __init__(self, beacon_pos, sigma_dist=0.02, sigma_bearing=0.02, name="lighthouse"):
		super().__init__(name=name, is_external=True, obs_dim=2)
		self.sigma_dist = sigma_dist
		self.sigma_bearing = sigma_bearing
		beacon_pos = np.array(beacon_pos)
		self.beacon_pos = beacon_pos
	def observe(self, state, action, rng):
		x, y, _ = state
		dx =  x-self.beacon_pos[0]
		dy =  y-self.beacon_pos[1]
		bearing = np.arctan2(dy, dx) + rng.normal(0, self.sigma_bearing)
		distance = np.hypot(dx, dy) + rng.normal(0, self.sigma_dist)
		bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
		return np.array([distance, bearing])