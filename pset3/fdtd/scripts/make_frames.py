import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = sys.argv
    data_file = args[1]
    save_dir = args[2]
    try:
        field = args[3]
    except IndexError:
        field = "Hzx"

    data = np.load(data_file)[field]
    vmin = data.min()
    vmax = data.max()
    basename = os.path.splitext(os.path.split(data_file)[1])[0]
    for i in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(5,5), dpi=150)
        cax = ax.imshow(data[i].T, aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(cax)
        filename = os.path.join(save_dir, f"{basename}_frame_{i}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
