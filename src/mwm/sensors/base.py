class Sensor:
	def __init__(self, name, is_external, obs_dim):
		self.name = name
		self.is_external = is_external
		self.obs_dim = obs_dim
	def observe(self, state, action, rng):
		raise NotImplementedError(
            "Subclasses must implement observe()"
        )
	def reset(self , state):
		pass