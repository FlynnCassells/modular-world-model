import numpy as np
import json
import torch
from torch.utils.data import Dataset

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
        # convert each array to a float32 tensor, once
        # build self.index: a list of (episode, start) pairs
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        # look up (episode, start)
        # return a dict of that window:
        #   obs per sensor:  steps start .. start + window_len
        #   actions:         same range
        #   states:          start .. start + window_len + 1  (one longer)
        ...