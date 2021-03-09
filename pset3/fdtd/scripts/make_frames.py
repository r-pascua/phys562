import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = sys.argv
    data_file = args[1]
    save_dir = args[2]

    data = np.load(data_file, allow_pickle=True)
    Ex = data["Ex"]
    Ey = data["Ey"]
    Hz = data["Hzx"] + data["Hzy"]
    labels = ("Ex", "Ey", "Hz")
    basename = os.path.splitext(os.path.split(data_file)[1])[0]
    for i in range(1, data["Ntimes"]):
        fig, axes = plt.subplots(1, 3, figsize=(15,5), dpi=150)
        for ax, field in zip(axes, (Ex, Ey, Hz)):
            ax.imshow(
                np.abs(field[i]).T / np.abs(field[i]).max(),
                aspect='auto',
                origin="lower",
                vmin=0,
                vmax=1
            )
        filename = os.path.join(save_dir, f"{basename}_frame_{i}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
