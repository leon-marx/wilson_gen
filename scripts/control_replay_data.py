# Sometimes, stablediffusion generates corrupted images.
# This file checks, that all images can be loaded.

import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# REPLAY_ROOT = "replay_data_cap"
REPLAY_ROOT = input("REPLAY_ROOT: ")
OV = input("OV [ov/dj/all]]: ")
TASK = "10-10"
if OV == "ov":
    ov_list = ["-ov"]
elif OV == "dj":
    ov_list = [""]
else:
    ov_list = ["", "-ov"]

for OV in ov_list:
    print(f"Checking {REPLAY_ROOT}/{TASK}{OV}")
    img_shapes = []
    psl_shapes = []
    img_pxl_vals = []
    psl_pxl_vals = []
    os.chdir(f"/home/thesis/marx/wilson_gen/WILSON/{REPLAY_ROOT}/{TASK}{OV}")
    corr_imgs = []
    corr_psls = []
    for dir in tqdm(os.listdir(), leave=True):
        if os.path.isdir(dir):
            for img in tqdm(os.listdir(f"{dir}/images"), leave=False):
                try:
                    imgg = Image.open(f"{dir}/images/{img}").convert("RGB")
                    img_shapes.append(pil_to_tensor(imgg).shape)
                    img_pxl_vals.append(imgg.load()[0, 0])
                except OSError:
                    corr_imgs.append(f"{dir}_{img}")
                try:
                    psl = Image.open(f"{dir}/pseudolabels/{img[:-4]}.png")
                    psl_shapes.append(pil_to_tensor(psl).shape)
                    psl_pxl_vals.append(psl.load()[0, 0])
                except OSError:
                    corr_psls.append(f"{dir}_{img}")
    if corr_imgs == [] and corr_psls == []:
        print(f"All images can be loaded for {REPLAY_ROOT}/{TASK}{OV}")
    else:
        print("Found the following corrupted images")
        for c_img in corr_imgs:
            print(c_img)
        print("Found the following corrupted pseudolabels")
        for c_psl in corr_psls:
            print(c_psl)
