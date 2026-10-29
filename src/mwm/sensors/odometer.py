import numpy as np
from mwm.sensors.base import Sensor

class OdometrySensor(Sensor):
	def __init__(self, sigma_left=0.0, sigma_right=0.0, scale_factor=1.01, name="odometer", wheelbase=0.3):
		super().__init__(name=name, is_external=False, obs_dim=2)
		self.sigma_left = sigma_left
		self.sigma_right = sigma_right
		self.scale_factor = scale_factor
		self.wheelbase = wheelbase
	def observe(self, state, last_executed, rng):

		dL = (last_executed[0] - last_executed[1]*self.wheelbase/2)
		dL += rng.normal(0, self.sigma_left*abs(dL))
		dR = self.scale_factor*(last_executed[0] + last_executed[1]*self.wheelbase/2)
		dR += rng.normal(0, self.sigma_right*abs(dR))
		return np.array([dL, dR])
