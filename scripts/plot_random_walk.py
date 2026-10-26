import numpy as np
import matplotlib.pyplot as plt
from mwm.rollout import rollout
from mwm.policies import generate_action, wall_aware_action
from mwm.envs.arena import Arena2D
from mwm.sensors.lighthouse import LighthouseSensor

seed = 0;
N_steps = 500;
margin = 0.3


def plot_walk(positions, arena):
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
		walks.append(rollout(N_steps, arena, i))
	positions = np.concatenate(walks)
	plt.hist2d(positions[:,0], positions[:,1], bins=20, range=[[0, arena.size], [0, arena.size]])
	plt.gca().set_aspect("equal")
	plt.colorbar()
	plt.show()

def plot_sensor(arena):
	positions, observations = rollout(N_steps, arena, lighthouse)
	fig, axes = plt.subplots(3,1)
	true_dists = np.hypot(positions[:,0], positions[:,1]-5.0)
	true_bearings = np.arctan2(positions[:,1]-lighthouse.beacon_pos[1], positions[:,0]-lighthouse.beacon_pos[0])
	t = np.arange(N_steps) * arena.dt
	axes[0].plot(t, true_dists, label="true")
	axes[0].plot(t, observations[:,0], label="sensor")
	axes[0].set_ylabel("distance (m)")
	axes[0].legend()
	axes[1].plot(t, true_bearings, label="true")
	axes[1].plot(t, observations[:,1], label="sensor")
	axes[1].set_ylabel("bearings (rad)")
	axes[1].legend()
	sensor_x = lighthouse.beacon_pos[0]+observations[:,0]*np.cos(observations[:,1])
	sensor_y = lighthouse.beacon_pos[1]+observations[:,0]*np.sin(observations[:,1])
	axes[2].plot(positions[:,0], positions[:,1], label="true")
	axes[2].plot(sensor_x, sensor_y, label="sensor")
	axes[2].legend()
	plt.show()


arena = Arena2D()
lighthouse = LighthouseSensor(beacon_pos=[0.0,5.0])
plot_sensor(arena)

