import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.insert(0, "/home/thesis/marx/wilson_gen/WILSON")
from WILSON import tasks


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


def get_image_names_and_onehots(task_and_ov, step):
    idxs_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{task_and_ov}/train-{step}.npy"
    idxs = np.load(idxs_path)
    with open("/home/thesis/marx/wilson_gen/WILSON/data/voc/splits/train_aug.txt") as f:
        lines = f.readlines()
        lines = [l.strip().split(" ")[0][1:] for l in lines]
        image_names = np.array(lines)[idxs]
    onehot_labels = np.load("/home/thesis/marx/wilson_gen/WILSON/data/voc/voc_1h_labels_train.npy")
    onehot_labels = onehot_labels[idxs]
    return image_names, onehot_labels


def get_cossim(voc_embeddings, gen_embeddings):
    voc_embeddings_normed = voc_embeddings / np.linalg.norm(voc_embeddings, axis=1)[:, np.newaxis]
    gen_embeddings_normed = gen_embeddings / np.linalg.norm(gen_embeddings, axis=1)[:, np.newaxis]
    return np.matmul(voc_embeddings_normed, gen_embeddings_normed.T)


def embed_all(step_classes, image_names, onehot_labels):
    for cl in step_classes:
        exist_flag = True
        if not os.path.exists(f"dino_embeds/voc/{task_and_ov}/{cl}.npy"):
            exist_flag = False
            break
        if exist_flag:
            print(f"Embeddings for current step classes already exist, skipping.")
            return None

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16,).to("cuda")
    base_path = "/home/thesis/marx/wilson_gen/WILSON/data/voc/"
    embeddings = {cl: [] for cl in np.unique(step_classes)}
    for i in tqdm(range(0, len(image_names), 16), desc="Getting embeddings"):
        with torch.no_grad():
            images = [Image.open(base_path + img_name).convert("RGB") for img_name in image_names[i:i+16]]
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(**inputs.to("cuda"))
            embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for j, cl in enumerate(step_classes):
                for emb, onehot in zip(embeds, onehot_labels[i:i+16]):
                    if onehot[j] == 1:
                        embeddings[cl].append(emb)
    os.makedirs(f"dino_embeds/voc/{task_and_ov}", exist_ok=True)
    for cl, embeds in embeddings.items():
        np.save(f"dino_embeds/voc/{task_and_ov}/{cl}.npy", np.array(embeds))


if __name__ == "__main__":
    os.chdir("/home/thesis/marx/wilson_gen/hugface")
    # TASK = input("Enter task: ")
    # OVERLAP = input("Overlap? (y/n): ") == "y"
    # TOP_K = int(input("Top k (0 for all): "))

    # call this function as PYTHONPATH=. python hugface/prepare_voc_lora_multistep.py 10-1 y 100 (for 10-1-ov task and top 100 images)
    TASK = sys.argv[1]
    OVERLAP = sys.argv[2] == "y"
    TOP_K = int(sys.argv[3])

    print(f"Preparing voc for lora with the following hyperparameters:")
    print(f"    TASK: {TASK}")
    print(f"    OVERLAP: {OVERLAP}")
    print(f"    TOP_K: {TOP_K}")

    summary_msgs = []
    for STEP in tasks.tasks["voc"][TASK].keys():
        # print(STEP)
        # print(tasks.tasks["voc"][TASK][STEP])

        task_and_ov = TASK + "-ov" if OVERLAP else TASK
        min_class = min(tasks.tasks["voc"][TASK][STEP])
        max_class = max(tasks.tasks["voc"][TASK][STEP])
        step_classes = np.array(classes)[max(min_class-1, 0):max_class]  # tasks considers 0 to be background, here 0 is first actual class
        # print(step_classes)


        # get images and classes
        image_names, onehot_labels = get_image_names_and_onehots(task_and_ov, step=STEP)
        onehot_labels = onehot_labels[:, max(min_class-1, 0):max_class]

        # get embeddings for task
        if TOP_K != 0:
            embed_all(step_classes, image_names, onehot_labels)

        # optionally prune images
        if TOP_K != 0:
            scores = {cl: [] for cl in step_classes}
            # scores[cl] wants list of tuples (img_name, onehot_label, score) for all images containing class cl
            for i, cl in tqdm(enumerate(step_classes), desc="Getting scores", leave=True):
                cl_embeds = np.load(f"dino_embeds/voc/{task_and_ov}/{cl}.npy")
                cl_ind = 0
                for j, img_name in enumerate(tqdm(image_names, desc=f"{cl}", leave=False)):
                    if onehot_labels[j, i] == 1:
                        embed = cl_embeds[cl_ind].reshape(1, -1)
                        scores[cl].append((img_name, onehot_labels[j], get_cossim(cl_embeds, embed).mean()))
                        cl_ind += 1

            image_names = []
            onehot_labels = []
            image_class_map = {}  # this is used to collect double entries for images that contain multiple classes
            confl_imgs = []
            for cl in step_classes:
                scores[cl] = sorted(scores[cl], key=lambda x: x[2], reverse=True)
                for img_name, onehot_label, score in scores[cl][:TOP_K]:
                    if img_name in image_class_map:
                        image_class_map[img_name].append((cl, score))
                        confl_imgs.append(img_name)
                    else:
                        image_names.append(img_name)
                        onehot_labels.append(onehot_label)
                        image_class_map[img_name] = [(cl, score)]
            conflict_class_inds = {cl: TOP_K for cl in step_classes}
            infinite_counter = 0
            while confl_imgs != []:  # add images if 2 classes claim the same image for themselves
                # for img_name, cl_tups in image_class_map.items():
                for img_name in confl_imgs:
                    cl_tups = sorted(image_class_map[img_name], key=lambda x: x[1], reverse=True)  # sort by score
                    for cl, score in cl_tups[1:]:  # for lower scoring classes, add another image
                        add_img_name, add_onehot_label, add_score = scores[cl][conflict_class_inds[cl]]
                        conflict_class_inds[cl] += 1
                        if add_img_name in image_class_map:  # backup image also preexisting, add new conflict
                            image_class_map[add_img_name].append((cl, add_score))
                            confl_imgs.append(add_img_name)
                        else:
                            image_names.append(add_img_name)
                            onehot_labels.append(add_onehot_label)
                            image_class_map[add_img_name] = [(cl, add_score)]
                        confl_imgs.remove(img_name)  # remove image once from conflict list
                        image_class_map[img_name].remove((cl, score))  # remove image from class map
                print(f"{infinite_counter = }")
                infinite_counter += 1
                if infinite_counter > 10:
                    print("Infinite loop detected, breaking out of while loop.")
                    break

            # copy images into lora dir
            os.makedirs(f"multistep_voc_lora_top_{TOP_K}_{task_and_ov}/{STEP}/train", exist_ok=True)
            for image_name in tqdm(image_names, desc="Copying images"):
                src_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}"
                dst_path = f"multistep_voc_lora_top_{TOP_K}_{task_and_ov}/{STEP}/train/{image_name.split('/')[1]}"
                os.system(f"cp {src_path} {dst_path}")

            # make metadata file
            with open(f"multistep_voc_lora_top_{TOP_K}_{task_and_ov}/{STEP}/train/metadata.csv", "w") as f:
                f.write("file_name,text\n")
                for i, image_name in enumerate(image_names):
                    present_classes = step_classes[onehot_labels[i] == 1]
                    num_present_classes = len(present_classes)
                    if num_present_classes == 1:
                        prompt = f"photo of a {present_classes[0]}"
                    else:
                        np.random.shuffle(present_classes)  # shuffle to avoid bias for first class
                        prompt = "photo of"
                        for pc in present_classes[:-1]:
                            prompt += " a " + pc + ","
                        prompt += " and a " + present_classes[-1]
                        prompt = prompt.replace(", and", " and")
                    prompt = f'"{prompt}"'
                    f.write(f"{image_name.split('/')[1]},{prompt.replace('_', ' ')}\n")
        # save all images
        else:
            # copy images into lora dir
            os.makedirs(f"multistep_voc_lora_{task_and_ov}/{STEP}/train", exist_ok=True)
            for image_name in tqdm(image_names, desc="Copying images"):
                src_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}"
                dst_path = f"multistep_voc_lora_{task_and_ov}/{STEP}/train/{image_name.split('/')[1]}"
                os.system(f"cp {src_path} {dst_path}")

            # make metadata file
            with open(f"multistep_voc_lora_{task_and_ov}/{STEP}/train/metadata.csv", "w") as f:
                f.write("file_name,text\n")
                for i, image_name in enumerate(image_names):
                    present_classes = step_classes[onehot_labels[i] == 1]
                    num_present_classes = len(present_classes)
                    if num_present_classes == 1:
                        prompt = f"photo of a {present_classes[0]}"
                    else:
                        np.random.shuffle(present_classes)  # shuffle to avoid bias for first class
                        prompt = "photo of"
                        for pc in present_classes[:-1]:
                            prompt += " a " + pc + ","
                        prompt += " and a " + present_classes[-1]
                        prompt = prompt.replace(", and", " and")
                    prompt = f'"{prompt}"'
                    f.write(f"{image_name.split('/')[1]},{prompt.replace('_', ' ')}\n")
        n_train_imgs = len(os.listdir(dst_path.split("train")[0] + "train")) - 1  # -1 for metadata file
        summary_msgs.append(f"Added {n_train_imgs} images to {dst_path.split('train')[0] + 'train'}")
    for msg in summary_msgs:
        print(msg)
    print("Done!")

