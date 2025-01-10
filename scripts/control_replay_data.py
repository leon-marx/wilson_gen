# Sometimes, stablediffusion generates corrupted images.
# This file checks, that all images can be loaded.

import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

REPLAY_ROOT = "replay_data_cap"
TASK = "10-10"

for OV in ["", "-ov"]:
    print(f"Checking {REPLAY_ROOT}/{TASK}{OV}")
    img_shapes = []
    os.chdir(f"/home/thesis/marx/wilson_gen/WILSON/{REPLAY_ROOT}/{TASK}{OV}")
    corr_imgs = []
    for dir in tqdm(os.listdir(), leave=True):
        for img in tqdm(os.listdir(f"{dir}/images"), leave=False):
            imgg = Image.open(f"{dir}/images/{img}")
            try:
                img_shapes.append(pil_to_tensor(imgg).shape)
            except OSError:
                corr_imgs.append(f"{dir}_{img}")
    if corr_imgs == []:
        print(f"All images can be loaded for {REPLAY_ROOT}/{TASK}{OV}")
    else:
        print("Found the following corrupted images")
        for c_img in corr_imgs:
            print(c_img)

    # DIRECT FIX OF BROKEN IMAGES (DOES NOT WORK ATM)
    # fix = input("Do you want to fix the broken images? (y/n)")
    # if fix == "y":
    #     GEN_DATA_ROOT = input("GEN_DATA_ROOT: ")
    #     os.chdir(f"/home/thesis/marx/wilson_gen/")

    #     classes = [
    #         "aeroplane",
    #         "bicycle",
    #         "bird",
    #         "boat",
    #         "bottle",
    #         "bus",
    #         "car",
    #         "cat",
    #         "chair",
    #         "cow",
    #         # "dining_table",
    #         # "dog",
    #         # "horse",
    #         # "motorbike",
    #         # "person",
    #         # "potted_plant",
    #         # "sheep",
    #         # "sofa",
    #         # "train",
    #         # "tv_monitor",
    #     ]

    #     for cl_name in classes:
    #         files = [f for f in os.listdir(f"stablediffusion/{GEN_DATA_ROOT}") if cl_name in f and ".png" in f]
    #         num_preexisting = len(os.listdir(f"WILSON/{REPLAY_ROOT}/{TASK + OV}/{cl_name}/images"))
    #         for ind, img in enumerate(files):
    #             if f"{cl_name}_{str(ind+len(files)).zfill(5)}.png" in corr_imgs:
    #                 print(f"ORIGINAL: stablediffusion/{GEN_DATA_ROOT}/{img}")
    #                 print(f"CORRUPTED: WILSON/{REPLAY_ROOT}/{TASK + OV}/{cl_name}/images/" + f"{str(ind+len(files)).zfill(5)}.png")
    #             # shutil.copy(f"{GEN_DATA_ROOT}/{img}", f"../WILSON/{REPLAY_ROOT}/{TASK_AND_OV}/{cl_name}/images/{str(ind+num_preexisting).zfill(5)}.png")
    # else:
    #     print("Exiting")
    #     exit()