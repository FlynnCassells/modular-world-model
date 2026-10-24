import numpy as np

class Arena2D:
	def __init__(self, size=10.0, dt=0.1, max_speed=1.0, max_turn=1.5):
		self.size= size
		self.dt = dt
		self.max_speed = max_speed
		self.max_turn = max_turn
	def reset(self, seed=None):
		self.rng = np.random.default_rng(seed)
		x = self.rng.uniform(1,9)
		y = self.rng.uniform(1,9)
		theta = self.rng.uniform(-np.pi,np.pi)
		self.state = [x, y, theta]
		return self.state;
	def step(self, action):
		v, omega = action
		x, y, theta = self.state
		#clip omega:
		omega = np.clip(omega, -self.max_turn, self.max_turn)
		theta += omega*self.dt
		theta = np.arctan2(np.sin(theta), np.cos(theta))
		#clip v
		v = np.clip(v, 0, self.max_speed)
		x += v*np.cos(theta)*self.dt
		y += v*np.sin(theta)*self.dt
		if(x<0): x = 0
		if(x>self.size): x = self.size
		if(y<0): y = 0
		if(y>self.size): y = self.size
		self.state = [x, y, theta]
		return self.state