from matplotlib import pyplot as plt
import numpy as np
import pickle

with open('data/others/motion_anchor_infos_mode6.pkl', 'rb') as f:
    datas = pickle.load(f)["anchors_all"]

x_max = -1
x_min = 100
y_max = -1
y_min = 100

for data in datas:
    x_max = max(np.max(data[:, :, 0]), x_max)
    x_min = min(np.min(data[:, :, 0]), x_min)
    y_max = max(np.max(data[:, :, 1]), y_max)
    y_min = min(np.min(data[:, :, 1]), y_min)

plt.xlim(x_min-1, x_max+1)
plt.ylim(y_min-1, y_max+1)
plt.grid()

for anchors in datas:
    for anchor in anchors:
        x = anchor[:, 0]
        y = anchor[:, 1]
        plt.plot(x, y, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="green")
plt.show()
