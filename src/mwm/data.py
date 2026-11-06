import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def load_dataset(name, folder="data"):
	with np.load(f"{folder}/{name}.npz") as f1:
		data = {key: f1[key] for key in f1.files}
	with open(f"{folder}/{name}.json") as f2:
		settings = json.load(f2)
	return data, settings

def split_dataset(arrays, splits):
	data_splits = {}
	for name, (start, end) in splits.items():
		data_splits[name] = {key: arrays[key][start:end] for key in arrays}
	return data_splits


def preprocess(arrays):
	d = arrays['obs_lighthouse'][...,0]
	bearing = arrays['obs_lighthouse'][...,1]
	lighthouse_features = np.stack([d, np.sin(bearing), np.cos(bearing)], axis=-1)
	new = dict(arrays)
	new['obs_lighthouse'] = lighthouse_features
	return new


def compute_norm_stats(train_arrays, keys):
	norm_stats = {}
	for key in keys:
		mean = np.mean(train_arrays[key], axis=(0,1))
		std = np.maximum(np.std(train_arrays[key], axis=(0,1)), 1e-6)
		norm_stats[key] = (mean, std)
	return norm_stats

def normalise(arrays, stats):
	norms = dict(arrays)
	for key, (mean, std) in stats.items():
		norms[key] = (arrays[key]-mean)/std
	return norms


class WindowDataset(Dataset):
	def __init__(self, arrays, window_len=50, stride=10):
		self.window_len = window_len
		self.obs = {}
		for key,array in arrays.items():
			tensor = torch.from_numpy(array).float()
			if key.startswith("obs_"):
				self.obs[key.removeprefix("obs_")] = tensor
			elif key =="actions":
				self.actions = tensor
			elif key == "states":
				self.states = tensor
		self.index = []
		n_episodes, n_steps = self.actions.shape[:2]
		for episode in range(n_episodes):
			for start in range(0, n_steps - window_len + 1, stride):
				self.index.append((episode, start))
	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		episode, start = self.index[idx]
		end = start + self.window_len
		dictionary = {}
		dictionary["obs"] = {key: data[episode, start:end] for key, data in self.obs.items()}
		dictionary["actions"] = self.actions[episode, start:end]
		dictionary["states"] = self.states[episode, start:end+1]
		return dictionary

def build_datasets(name, window_len=50, stride=10, folder="data"):
	arrays, settings = load_dataset(name, folder)
	arrays = preprocess(arrays)
	splits = split_dataset(arrays, settings["splits"])
	obs_keys = [k for k in arrays if k.startswith("obs_")]
	stats = compute_norm_stats(splits["train"], obs_keys+["actions"])
	datasets = {
		splits_name: WindowDataset(normalise(part, stats), window_len, stride) for splits_name, part in splits.items()
	}
	return datasets, stats, settings






