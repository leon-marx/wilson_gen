# Sometimes, stablediffusion generates corrupted images.
# This file checks, that all images can be loaded.

import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

REPLAY_ROOT = "replay_data_baseline"
TASK = "10-10"
OV = "-ov"  # or ""

for REPLAY_ROOT in ["replay_data_baseline", "replay_data"]:
    for OV in ["", "-ov"]:
        print(f"Checking {REPLAY_ROOT}/{TASK}{OV}")
        img_shapes = []
        os.chdir(f"/home/thesis/marx/wilson_gen/WILSON/{REPLAY_ROOT}/{TASK}{OV}")
        for dir in tqdm(os.listdir(), leave=True):
            for img in tqdm(os.listdir(f"{dir}/images"), leave=False):
                imgg = Image.open(f"{dir}/images/{img}")
                try:
                    img_shapes.append(pil_to_tensor(imgg).shape)
                except OSError:
                    print(" ")
                    print(" ")
                    print(" ")
                    print(" ")
                    print(" ")
                    print(dir, img)
                    raise TypeError
        print(f"All images can be loaded for {REPLAY_ROOT}/{TASK}{OV}")