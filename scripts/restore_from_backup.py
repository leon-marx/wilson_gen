import os
os.chdir("/home/thesis/marx/wilson_gen/WILSON")

DATA_ROOT = input("DATA_ROOT: ")
backup_root = DATA_ROOT + "_backup"

TASK = "10-10"

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    # "dining_table",
    # "dog",
    # "horse",
    # "motorbike",
    # "person",
    # "potted_plant",
    # "sheep",
    # "sofa",
    # "train",
    # "tv_monitor",
]

for OV in ["", "-ov"]:
    for cl_dir in sorted(os.listdir(f"/home/thesis/marx/wilson_gen/WILSON/{DATA_ROOT}/{TASK + OV}")):
        if cl_dir in classes:
            print(f"Restoring {DATA_ROOT}/{TASK + OV}/{cl_dir}")
            os.system(f"rm -rf /home/thesis/marx/wilson_gen/WILSON/{DATA_ROOT}/{TASK + OV}/{cl_dir}/images")
            os.system(f"cp -r /home/thesis/marx/wilson_gen/WILSON/{backup_root}/{TASK + OV}/{cl_dir}/images /home/thesis/marx/wilson_gen/WILSON/{DATA_ROOT}/{TASK + OV}/{cl_dir}")
