import os
import numpy as np
from PIL import Image
from tqdm import tqdm


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
    "dining_table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted_plant",
    "sheep",
    "sofa",
    "train",
    "tv_monitor",
]


def get_image_names(task_and_ov):
    idxs_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{task_and_ov}/train-0.npy"
    idxs = np.load(idxs_path)
    with open("/home/thesis/marx/wilson_gen/WILSON/data/voc/splits/train_aug.txt") as f:
        lines = f.readlines()
        lines = [l.strip().split(" ")[0][1:] for l in lines]
        image_names = np.array(lines)[idxs]
    return image_names


def get_class(image_name):
    # new method, using centrality
    labels = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/SegmentationClassAug/{image_name.split('/')[1][:-4]}.png")
    w, h = labels.size
    labels = np.array(labels)
    uniques = np.unique(labels)
    uniques = [unq for unq in uniques if unq > 0 and unq <= len(classes)]
    min_dist = 1000000.0
    central_ind = None
    for unq in uniques:
        unq_dists = []
        for i in range(h):
            for j in range(w):
                if labels[i, j] == unq:
                    unq_dists.append(np.sqrt((i - h/2)**2 + (j - w/2)**2))
        if np.mean(unq_dists) < min_dist:
            min_dist = np.mean(unq_dists)
            central_ind = unq
    central_class = classes[central_ind - 1]
    return central_class


def get_cossim(voc_embeddings, gen_embeddings):
    voc_embeddings_normed = voc_embeddings / np.linalg.norm(voc_embeddings, axis=1)[:, np.newaxis]
    gen_embeddings_normed = gen_embeddings / np.linalg.norm(gen_embeddings, axis=1)[:, np.newaxis]
    return np.matmul(voc_embeddings_normed, gen_embeddings_normed.T)


if __name__ == "__main__":
    os.chdir("/home/thesis/marx/wilson_gen/hugface")
    TASK = input("Enter task: ")
    OVERLAP = input("Overlap? (y/n): ") == "y"
    TOP_K = int(input("Top k (0 for all): "))

    task_and_ov = TASK + "-ov" if OVERLAP else TASK
    classes = classes[:int(TASK.split("-")[0])]

    # get images and classes
    image_names = get_image_names(task_and_ov)
    image_classes = []
    for image_name in tqdm(image_names, desc="Getting classes"):
        image_classes.append(get_class(image_name))

    # optionally prune images
    if TOP_K != 0:
        scores = {cl: [] for cl in classes}
        classes = [cl[:-4] for cl in sorted(os.listdir(f"dino_embeds/voc/{task_and_ov}"))]
        embeddings = {cl: np.load(f"dino_embeds/voc/{task_and_ov}/{cl}.npy") for cl in classes}
        cl_inds = {cl: 0 for cl in classes}
        for img_name, cl in tqdm(list(zip(image_names, image_classes)), desc="Getting scores"):
            ind = cl_inds[cl]
            embed = embeddings[cl][ind].reshape(1, -1)
            scores[cl].append((img_name, get_cossim(embeddings[cl], embed).mean()))
            cl_inds[cl] += 1
        image_names = []
        image_classes = []
        for cl in classes:
            scores[cl] = sorted(scores[cl], key=lambda x: x[1], reverse=True)
            for img_name, _ in scores[cl][:TOP_K]:
                image_names.append(img_name)
                image_classes.append(cl)

        # copy images into lora dir
        os.makedirs(f"voc_lora_top_{TOP_K}_{task_and_ov}/train", exist_ok=True)
        for image_name in tqdm(image_names, desc="Copying images"):
            src_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}"
            dst_path = f"voc_lora_top_{TOP_K}_{task_and_ov}/train/{image_name.split('/')[1]}"
            os.system(f"cp {src_path} {dst_path}")

        # make metadata file
        with open(f"voc_lora_top_{TOP_K}_{task_and_ov}/train/metadata.csv", "w") as f:
            f.write("file_name,text\n")
            for i, image_name in enumerate(image_names):
                f.write(f"{image_name.split('/')[1]},photo of a {image_classes[i]}\n")
        print("Done!")
    else:
        # copy images into lora dir
        os.makedirs(f"voc_lora_{task_and_ov}/train", exist_ok=True)
        for image_name in tqdm(image_names, desc="Copying images"):
            src_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}"
            dst_path = f"voc_lora_{task_and_ov}/train/{image_name.split('/')[1]}"
            os.system(f"cp {src_path} {dst_path}")

        # make metadata file
        with open(f"voc_lora_{task_and_ov}/train/metadata.csv", "w") as f:
            f.write("file_name,text\n")
            for i, image_name in enumerate(image_names):
                f.write(f"{image_name.split('/')[1]},photo of a {image_classes[i]}\n")
        print("Done!")

