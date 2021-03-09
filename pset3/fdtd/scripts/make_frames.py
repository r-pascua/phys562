import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants

if __name__ == "__main__":
    args = sys.argv
    data_file = args[1]
    save_dir = args[2]

    data = np.load(data_file, allow_pickle=True)
    Ex = data["Ex"]
    Ex = 0.5 * (Ex[:,:,1:] + Ex[:,:,:-1])
    Ey = data["Ey"]
    Ey = 0.5 * (Ey[:,1:,:] + Ey[:,:-1,:])
    Hz = data["Hzx"] + data["Hzy"]
    permeability = data["relative_permeability"]
    impedance = np.sqrt(constants.mu0.value / constants.eps0.value)
    labels = ("Ex", "Ey", "Hz")
    basename = os.path.splitext(os.path.split(data_file)[1])[0]
    for i in range(1, data["Ntimes"]):
        if (100 * i / data["Ntimes"]) % 10 == 0:
            print(f"{100 * i / data['Ntimes']}% done.")
        fig, axes = plt.subplots(1, 3, figsize=(15,5), dpi=150)
        for ax, field, name in zip(axes, (Ex, Ey, Hz), ("Ex", "Ey", "Hz")):
            ax.set_title(name)
            if name == "Hz":
                scale = np.abs(field[i]).max()
            else:
                scale = impedance * permeability * np.abs(Hz[i]).max()
            ax.imshow(
                np.abs(field[i]).T / scale,
                aspect='auto',
                origin="lower",
                vmin=0,
                vmax=1
            )
        filename = os.path.join(save_dir, f"{basename}_frame_{i}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
