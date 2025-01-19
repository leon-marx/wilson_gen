import os
from PIL import Image
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
    print(f"Converting {REPLAY_ROOT}/{TASK}{OV}")
    os.chdir(f"/home/thesis/marx/wilson_gen/WILSON/{REPLAY_ROOT}/{TASK}{OV}")
    for dir in tqdm(os.listdir(), leave=True):
        if os.path.isdir(dir):
            for image_name in tqdm(sorted(os.listdir(f"{dir}/images")), leave=False):
                if image_name.endswith(".png"):
                    try:
                        im = Image.open(f"{dir}/images/{image_name}")
                        im.convert("RGB").save(f"{dir}/images/{image_name[:-4]}.jpg", "JPEG", quality=95, optimize=True)
                        os.system(f"rm {dir}/images/{image_name}")
                    except Exception as e:
                        print(f"Failed to convert {dir}/images/{image_name} because of:")
                        print(e)