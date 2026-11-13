import matplotlib.pyplot as plt
import json
import numpy

if __name__=="__main__":
	with open(f"runs/20261112_134455/history.json") as f:
		data = json.load(f)
	epoches = [e["epoch"] for e in data["epochs"]]
	lh_ratio = [e["baseline_ratio"]["lighthouse"] for e in data["epochs"]]
	lh_train = [e["train"]["lighthouse"] for e in data["epochs"]]
	lh_val = [e["val"]["lighthouse"] for e in data["epochs"]]
	odo_ratio = [e["baseline_ratio"]["odometer"] for e in data["epochs"]]
	odo_train = [e["train"]["odometer"] for e in data["epochs"]]
	odo_val = [e["val"]["odometer"] for e in data["epochs"]]
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
	axes[0,0].plot(epoches, lh_ratio)
	axes[0,0].set_title("Lighthouse Baseline Ratio")

	axes[0,1].plot(epoches, lh_val)
	axes[0,1].set_title("Lighthouse Validation Loss")
	axes[0,1].set_yscale('log')

	axes[0,2].plot(epoches, lh_train)
	axes[0,2].set_title("Lighthouse Training Loss")
	axes[0,2].set_yscale('log')

	axes[1,0].plot(epoches, odo_ratio)
	axes[1,0].set_title("Odometer Baseline Ratio")

	axes[1,1].plot(epoches, odo_val)
	axes[1,1].set_title("Odometer Validation Loss")
	axes[1,1].set_yscale('log')

	axes[1,2].plot(epoches, odo_train)
	axes[1,2].set_title("Odometer Traing Loss")
	axes[1,2].set_yscale('log')
	plt.show()

