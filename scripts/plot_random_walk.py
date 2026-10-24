import numpy as np
import matplotlib.pyplot as plt
from mwm.envs.arena import Arena2D

seed = 0;
N_steps = 500;
margin = 0.3

def generate_action(arena, rng):
	v = rng.uniform(0,arena.max_speed)
	omega = rng.uniform(-arena.max_turn, arena.max_turn)
	return np.array([v, omega])

def wall_aware_action(arena, rng):
	x, y, theta = arena.state
	if(margin<x<arena.size-margin and margin<y<arena.size-margin):
		return generate_action(arena, rng)
	else:
		dx = arena.size/2 - x
		dy = arena.size/2 - y
		center_heading = np.arctan2(dy, dx)
		error = np.arctan2(np.sin(center_heading - theta), np.cos(center_heading - theta))
		omega = np.sign(error)*arena.max_turn
		v, _ = generate_action(arena, rng)
		return np.array([v, omega])


def walk(N_steps, arena, seed=None):
	rng = np.random.default_rng(seed)
	arena.reset(seed)
	xs = np.zeros(N_steps)
	ys = np.zeros(N_steps)
	thetas = np.zeros(N_steps)
	for i in range(0,N_steps):
		x, y, theta = arena.step(wall_aware_action(arena, rng));
		xs[i] = x;
		ys[i] = y;
		thetas[i] = theta;
	return np.stack([xs, ys, thetas], axis=1)

def plot(positions, arena):
	plt.plot(positions[:,0], positions[:, 1])
	plt.scatter(positions[0,0], positions[0,1], color="blue")
	plt.scatter(positions[-1,0], positions[-1,1], color="red")
	plt.axis([0, arena.size, 0, arena.size])
	plt.gca().set_aspect("equal")
	
	plt.savefig("runs/random_walk.png")
	plt.show()

def check_coverage(arena):
	walks = []
	for i in range(100):
		walks.append(walk(N_steps, arena, i))
	positions = np.concatenate(walks)
	plt.hist2d(positions[:,0], positions[:,1], bins=20, range=[[0, arena.size], [0, arena.size]])
	plt.gca().set_aspect("equal")
	plt.colorbar()
	plt.show()

arena = Arena2D()
check_coverage(arena)


