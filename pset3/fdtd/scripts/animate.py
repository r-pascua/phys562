import glob
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    args = sys.argv
    file_dir = args[1]
    save_dir = str(Path(file_dir).parent)
    save_filename = args[2]
    skip_frames = int(args[3])
    try:
        time_scale = float(args[4])
    except IndexError:
        time_scale = 1

    files = glob.glob(os.path.join(file_dir, "*.png"))
    files = sorted(files, key=lambda f: int(f[f.rindex("_")+1:-4]))
    frames = []
    for fn in files[::skip_frames]:
        frames.append(Image.open(fn))

    frames[0].save(
        os.path.join(save_dir, save_filename),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=time_scale * len(files[::skip_frames]),
        loop=0,
    )
