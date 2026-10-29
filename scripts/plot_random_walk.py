import numpy as np
import matplotlib.pyplot as plt
from mwm.rollout import rollout
from mwm.policies import generate_action, wall_aware_action
from mwm.envs.arena import Arena2D
from mwm.sensors.lighthouse import LighthouseSensor
from mwm.sensors.odometer import OdometrySensor

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

def dead_reckoning(dL, dR, initial_pos):
	d = (dL+dR)/2
	d_theta = (dR-dL)/odometer.wheelbase
	theta = initial_pos[2] + np.cumsum(d_theta)
	x = initial_pos[0] + np.cumsum(d*np.cos(theta))
	y = initial_pos[1] + np.cumsum(d*np.sin(theta))
	return x, y, theta

def plot_sensor(arena):
	positions, observations , _= rollout(N_steps, arena, seed=0, sensors=[lighthouse, odometer])
	fig, axes = plt.subplots(2,1)
	true_dists = np.hypot(positions[:,0]-lighthouse.beacon_pos[0], positions[:,1]-lighthouse.beacon_pos[1])
	true_bearings = np.arctan2(positions[:,1]-lighthouse.beacon_pos[1], positions[:,0]-lighthouse.beacon_pos[0])
	odo_x, odo_y, odo_theta = dead_reckoning(observations[odometer.name][:,0], observations[odometer.name][:,1], positions[0,:])
	odo_distances = np.hypot(odo_x-lighthouse.beacon_pos[0], odo_y-lighthouse.beacon_pos[1])
	t = np.arange(N_steps+1) * arena.dt
	axes[0].plot(t, true_dists, label="true")
	axes[0].plot(t[1:], observations[lighthouse.name][:,0], label="external sensor")
	axes[0].plot(t[1:], odo_distances, label="internal sensor")
	axes[0].set_ylabel("distance (m)")
	axes[0].legend()
	axes[1].plot(t, true_bearings, label="true")
	axes[1].plot(t[1:], observations[lighthouse.name][:,1], label="external sensor")
	#axes[1].plot(t, odo_theta, label="internal sensor")
	axes[1].set_ylabel("bearings (rad)")
	axes[1].legend()
	sensor_x = lighthouse.beacon_pos[0]+observations[lighthouse.name][:,0]*np.cos(observations[lighthouse.name][:,1])
	sensor_y = lighthouse.beacon_pos[1]+observations[lighthouse.name][:,0]*np.sin(observations[lighthouse.name][:,1])
	fig2, axes2 = plt.subplots(1,2)
	axes2[0].plot(positions[:,0], positions[:,1], label="true")
	axes2[0].plot(sensor_x, sensor_y, label="external sensor")
	axes2[0].plot(odo_x, odo_y, label="internal sensor")
	for pos in positions:
		if pos[0]<=0 or pos[0]>=10 or pos[1]<=0 or pos[1]>=10:
			axes2[0].scatter(pos[0], pos[1], color='red')
	axes2[0].axis([0, arena.size, 0, arena.size])
	axes2[0].set_aspect("equal")
	axes2[0].legend()
	error = np.hypot(positions[1:,0]-odo_x, positions[1:,1]-odo_y)
	axes2[1].plot(t[1:], error)
	plt.show()

if __name__ =='__main__':
	arena = Arena2D()
	lighthouse = LighthouseSensor(beacon_pos=[0.0,5.0])
	odometer = OdometrySensor(scale_factor=1.0)
	plot_sensor(arena)

