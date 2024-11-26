import os

path = "./replay_data/10-10"
dirs = sorted(os.listdir(path))
for dir in dirs:
    print(dir)
    file_path = os.path.join(path, dir)
    img_path = os.path.join(file_path, "image")
    label_path = os.path.join(file_path, "label")
    # label_path = os.path.join(file_path, "label")
    img_files = os.listdir(img_path)

    with open(os.path.join(file_path, "train_fullPath.txt"), "w+") as f:
        for idx in range(len(img_files)):
            img_file = os.path.join("image", img_files[idx])
            img_file_fullPath = os.path.join(img_path, img_files[idx] )
            label_file_fullPath = os.path.join(label_path, img_files[idx][:-3]+"png" )
            label_file = os.path.join("label", img_files[idx][:-3]+"png")
            f.write(img_file_fullPath.replace("\\", "/") + " " + label_file_fullPath.replace("\\", "/") + "\n")
    print("Finished writing train file.")
