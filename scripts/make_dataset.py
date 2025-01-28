import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from omegaconf import OmegaConf
from einops import rearrange
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import pil_to_tensor
from imwatermark import WatermarkEncoder
from itertools import islice
import contextlib
import matplotlib.pyplot as plt
import json
import cv2
import pickle
import sys
sys.path.insert(0, "/home/thesis/marx/wilson_gen/stablediffusion")
from stablediffusion.ldm.util import instantiate_from_config
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
sys.path.insert(0, "/home/thesis/marx/wilson_gen/WILSON")
from WILSON.segmentation_module import make_model
from WILSON import tasks
from WILSON.dataset import transform

torch.set_grad_enabled(False)

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


_transform = transform.Compose([
    # transform.Resize(size=512),  # Not necessary, gen_imgs are 512x512 already
    # transform.CenterCrop(size=512),
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])


class wilson_opts_obj():
    def __init__(self, task, overlap):
        self.dataset = "voc"
        self.task = task
        self.step = 0
        self.test = True
        self.norm_act = "iabn_sync"
        self.backbone = "resnet101"
        self.output_stride = 16
        self.no_pretrained = False
        self.pooling = 32
        if overlap:
            self.ckpt = f"/home/thesis/marx/wilson_gen/WILSON/checkpoints/step/voc-{task}-ov/Base_0.pth"
        else:
            self.ckpt = f"/home/thesis/marx/wilson_gen/WILSON/checkpoints/step/voc-{task}/Base_0.pth"


def get_image_names(task_and_ov):
    idxs_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{task_and_ov}/train-0.npy"
    idxs = np.load(idxs_path)
    with open("/home/thesis/marx/wilson_gen/WILSON/data/voc/splits/train_aug.txt") as f:
        lines = f.readlines()
        lines = [l.strip().split(" ")[0][1:] for l in lines]
        image_names = np.array(lines)[idxs]
    return image_names


def load_caption_model():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).eval().to("cuda")
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True
        )
    return model, processor


def load_diffusion_model(config):
    ckpt = "/home/thesis/marx/wilson_gen/stablediffusion/pretrained/v2-1_512-ema-pruned.ckpt"
    print(f"Loading model from {ckpt}")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model = model.eval().to("cuda")
    return model


def load_pseudolabel_model(opts):
    print(f"Loading checkpoint from {opts.ckpt}")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        os.chdir("/home/thesis/marx/wilson_gen/WILSON")
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        net_dict = model.state_dict()
        pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items() if
            (k.replace("module.", "") in net_dict) and
            (v.shape == net_dict[k.replace("module.", "")].shape)}
        assert pretrained_dict.keys() == net_dict.keys(), "STATE DICTS NOT COMPATIBLE!"
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict)
        del checkpoint
        model = model.to("cuda")
        for par in model.parameters():
                par.requires_grad = False
        model.eval()
        os.chdir("/home/thesis/marx/wilson_gen")
    return model


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


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

    # old method, using class counts
    # labels = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/SegmentationClassAug/{image_name.split('/')[1][:-4]}.png")
    # labels = np.array(labels)
    # uniques, counts = np.unique(labels, return_counts=True)
    # if 0 in uniques:
    #     uniques = uniques[1:]
    #     counts = counts[1:]
    # uniques = [unq for unq in uniques if unq <= 10]
    # counts = counts[:len(uniques)]
    # return classes[uniques[np.argmax(counts)]-1]


def get_captions(replay_root, task_and_ov, image_names, prompt, repeat, img_gen_batch_size):
    fix_flag = False
    if prompt == "<CAPTION_FIX>":
        prompt = "<CAPTION>"
        fix_flag = True
    model, processor = load_caption_model()
    with torch.no_grad():
        print("Generating captions:")
        metadata = {}
        captions = []
        class_counts = {c: 0 for c in classes}
        for image_name in tqdm(image_names):
            image_class = get_class(image_name)
            if prompt == "<CLASSES>":
                prompt = f"What does the image of a {cl} describe?"
            image = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}").convert("RGB")
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float32)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=prompt)
            caption = parsed_answer[prompt].replace("\n", " ")
            if fix_flag:
                caption = f"A photo of a {image_class}: " + caption
            captions += [caption] * repeat
            metadata[image_name] = {"caption": caption}
            metadata[image_name]["class"] = image_class
            metadata[image_name]["voc_name"] = image_name
            metadata[image_name]["gen_names"] = []
            class_counts[metadata[image_name]["class"]] += 1
        del model
        del processor
        torch.cuda.empty_cache()
        if len(captions) % img_gen_batch_size != 0:  # padding empty captions to fit batch size
            for i in range(img_gen_batch_size - len(captions) % img_gen_batch_size):
                captions.append("")
                metadata[f"DELETEME_{i}"] = {"caption": "", "class": "DELETEME", "voc_name": "", "gen_names": []}

        # saving class counts file
        with open(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/class_counts.pkl", "wb") as f:
            pickle.dump(class_counts, f)

        return captions, metadata
    # with open(f"/home/thesis/marx/wilson_gen/{replay_root}/{task_and_ov}/voc_captions.txt", "w") as f:
    #     for caption in captions:
    #         caption = caption.replace("\n", " ")
    #         f.write(caption + "\n")
    #     if len(captions) % img_gen_batch_size != 0:
    #         for i in range(img_gen_batch_size - len(captions) % img_gen_batch_size):
    #             f.write("\n")


def gen_images(replay_root, captions, metadata, repeat, img_gen_batch_size):
    config = OmegaConf.load("/home/thesis/marx/wilson_gen/stablediffusion/configs/stable-diffusion/v2-inference.yaml")
    device = torch.device("cuda")
    model = load_diffusion_model(config)

    sampler = DDIMSampler(model, device=device)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    data = list(chunk(captions, img_gen_batch_size))
    image_names = list(metadata.keys())


    with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
        class_counts = {cl: 0 for cl in classes}
        global_ind = 0
        print("Generating images:")
        for prompts in tqdm(data, desc="data"):
            uc = model.get_learned_conditioning(img_gen_batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [4, 64, 64]
            samples, _ = sampler.sample(
                S=50,
                conditioning=c,
                batch_size=img_gen_batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.5,
                x_T=None
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for i, x_sample in enumerate(x_samples):
                img_class = metadata[image_names[(i + global_ind) // repeat]]["class"]
                if img_class != "DELETEME":
                    img_number = class_counts[img_class]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(f"WILSON/{replay_root}/{task_and_ov}/{img_class}/images/{img_number:05}.jpg", "JPEG", quality=95, optimize=True)
                    metadata[image_names[(i + global_ind) // repeat]]["gen_names"].append(f"{task_and_ov}/{img_class}/images/{img_number:05}.jpg")
                    class_counts[img_class] += 1
            global_ind += img_gen_batch_size
        del model
        del sampler
        torch.cuda.empty_cache()
        bad_keys = []
        for k in metadata:
            if "DELETEME" in k:
                bad_keys.append(k)
        for k in bad_keys:
            del metadata[k]


def generate_labels(replay_root, task_and_ov, opts):
    bad_images = []
    with torch.no_grad():
        pseudo_labeler = load_pseudolabel_model(opts)
        print("Generating pseudolabels:")
        for cf in tqdm(classes, leave=True):
            pseudolabels_1h = dict()
            img_names = os.listdir(os.path.join(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}", cf, "images"))
            for img_name in tqdm(img_names, leave=False):
                img = Image.open(os.path.join(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}", cf, "images", img_name)).convert("RGB")
                img = _transform(img).to("cuda", dtype=torch.float32).unsqueeze(0)
                outputs = pseudo_labeler(img)[0].cpu().numpy().squeeze()
                # print(outputs.shape)
                # print(outputs)
                pseudo_label = np.argmax(outputs, axis=0).astype(np.uint8)
                # print(pseudo_label.shape)
                # print(pseudo_label)
                pseudo_label_1h = np.zeros((21), dtype=np.uint8)
                for present_class in np.unique(pseudo_label):
                    if present_class <= 10:
                        pseudo_label_1h[int(present_class)] = 1
                if pseudo_label_1h[classes.index(cf)+1] != 1:
                    # print(f"Class {cf} not present in image {cf}/images/{img_name}")
                    bad_images.append(f"{task_and_ov}/{cf}/images/{img_name}")
                pseudolabels_1h[f"{img_name[:-4]}.png"] = pseudo_label_1h[1:] # Exclude background class
                pseudo_label = Image.fromarray(pseudo_label)
                pseudo_label.save(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/{cf}/pseudolabels/{img_name[:-4]}.png")
            with open(os.path.join(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}", cf, "pseudolabels_1h.pkl"), "wb") as f:
                pickle.dump(pseudolabels_1h, f)
        del pseudo_labeler
        torch.cuda.empty_cache()
    if bad_images == []:
        os.system(f"rm -rf /home/thesis/marx/wilson_gen/WILSON/{replay_root}_bad")
        print("All images have the correct class present")
    return bad_images


def control_replay_data(replay_root, task_and_ov, metadata):
    img_shapes = []
    psl_shapes = []
    img_pxl_vals = []
    psl_pxl_vals = []
    corr_imgs = []
    corr_psls = []
    print("Controlling images:")
    for voc_img in tqdm(metadata, leave=True):
        for gen_img in metadata[voc_img]["gen_names"]:
            try:
                imgg = Image.open(f"WILSON/{replay_root}/{gen_img}").convert("RGB")
                img_shapes.append(pil_to_tensor(imgg).shape)
                img_pxl_vals.append(imgg.load()[0, 0])
            except OSError:
                corr_imgs.append(f"{gen_img}")
            try:
                psl = Image.open(f"WILSON/{replay_root}/{gen_img.replace('/images/', '/pseudolabels/')[:-4]}.png")
                psl_shapes.append(pil_to_tensor(psl).shape)
                psl_pxl_vals.append(psl.load()[0, 0])
            except OSError:
                corr_psls.append(f"{gen_img}")
    if corr_imgs == [] and corr_psls == []:
        print(f"All images can be loaded for {replay_root}/{task_and_ov}")
    else:
        print("Found the following corrupted images")
        for c_img in corr_imgs:
            print(c_img)
        print("Found the following corrupted pseudolabels")
        for c_psl in corr_psls:
            print(c_psl)


def post_processing(replay_root, task_and_ov, metadata, repeat):
    with open(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/metadata.json", "w") as f:
        json.dump(metadata, f)
    class_to_imgs = {cl: [] for cl in classes}
    for d in list(metadata.values()):
        class_to_imgs[d["class"]].append(([d["voc_name"]] + d["gen_names"]))
    fig, axes = plt.subplots(2 * len(classes), repeat + 1, figsize=(5 * (repeat + 1), 10 * len(classes)))
    for i, cl in enumerate(classes):
        for j in range(2):
            voc_img_0 = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{class_to_imgs[cl][j][0]}").convert("RGB")
            axes[2 * i + j, 0].imshow(voc_img_0)
            axes[2 * i + j, 0].set_title(f"{class_to_imgs[cl][j][0]}")
            axes[2 * i + j, 0].set_axis_off()
            for k in range(repeat):
                gen_img = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{class_to_imgs[cl][j][1 + k]}").convert("RGB")
                axes[2 * i + j, 1 + k].imshow(gen_img)
                axes[2 * i + j, 1 + k].set_title(f"{class_to_imgs[cl][j][1 + k]}")
                axes[2 * i + j, 1 + k].set_axis_off()
    plt.tight_layout()
    plt.savefig(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/grid.png")
    plt.close(fig)


def repeat_bad_images_base(replay_root, task_and_ov, bad_images, prompt, repeat, img_gen_batch_size, metadata, seed, counter):
    replay_root_bad = replay_root + "_bad"
    for cl in classes:
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/{cl}/images", exist_ok=True)
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/{cl}/pseudolabels", exist_ok=True)
    seed_everything(seed+counter)
    bad_image_names = []
    for k, d in metadata.items():
        for gen_name in d["gen_names"]:
            if gen_name in bad_images:
                bad_image_names.append(k)
                break
    new_captions, new_metadata = get_captions(replay_root_bad, task_and_ov, bad_image_names, prompt, repeat, img_gen_batch_size)
    for i, bad_img in enumerate(bad_image_names):
        new_metadata[bad_img]["caption"] = f"photo of a {new_metadata[bad_img]['class']}"
        for j in range(repeat):
            new_captions[i * repeat + j] = f"photo of a {new_metadata[bad_img]['class']}"
    gen_images(replay_root_bad, new_captions, new_metadata, repeat, img_gen_batch_size)
    for bad_img in bad_image_names:
        new_names = new_metadata[bad_img]["gen_names"]
        old_names = metadata[bad_img]["gen_names"]
        for j in range(repeat):
            os.system(f"mv /home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{new_names[j]} /home/thesis/marx/wilson_gen/WILSON/{replay_root}/{old_names[j]}")
        metadata[bad_img]["caption"] = new_metadata[bad_img]["caption"]
    # os.system(f"rm -rf /home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}")

    for i, bad_img in enumerate(bad_image_names):
        fig, axes = plt.subplots(1, repeat + 1)
        voc_img = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{bad_img}").convert("RGB")
        axes[0].imshow(voc_img)
        axes[0].set_title(f"{bad_img}")
        axes[0].set_axis_off()
        for j in range(repeat):
            gen_img = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{metadata[bad_img]['gen_names'][j]}").convert("RGB")
            axes[1 + j].imshow(gen_img)
            axes[1 + j].set_title(f"{metadata[bad_img]['gen_names'][j]}")
            axes[1 + j].set_axis_off()
        fig.text(0.5, 0.04, f"{metadata[bad_img]['caption']}", ha='center')
        plt.tight_layout()
        plt.savefig(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/compare_{i}.png")
        plt.close(fig)

    return metadata


def repeat_bad_images(replay_root, task_and_ov, bad_images, prompt, repeat, img_gen_batch_size, metadata, seed, counter):
    replay_root_bad = replay_root + "_bad"
    for cl in classes:
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/{cl}/images", exist_ok=True)
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/{cl}/pseudolabels", exist_ok=True)
    seed_everything(seed+counter)
    bad_image_names = []
    for k, d in metadata.items():
        for gen_name in d["gen_names"]:
            if gen_name in bad_images:
                bad_image_names.append(k)
                break
    new_captions, new_metadata = get_captions(replay_root_bad, task_and_ov, bad_image_names, prompt, repeat, img_gen_batch_size)
    gen_images(replay_root_bad, new_captions, new_metadata, repeat, img_gen_batch_size)
    for bad_img in bad_image_names:
        new_names = new_metadata[bad_img]["gen_names"]
        old_names = metadata[bad_img]["gen_names"]
        for j in range(repeat):
            os.system(f"mv /home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{new_names[j]} /home/thesis/marx/wilson_gen/WILSON/{replay_root}/{old_names[j]}")
        metadata[bad_img]["caption"] = new_metadata[bad_img]["caption"]
    # os.system(f"rm -rf /home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}")

    for i, bad_img in enumerate(bad_image_names):
        fig, axes = plt.subplots(1, repeat + 1)
        voc_img = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{bad_img}").convert("RGB")
        axes[0].imshow(voc_img)
        axes[0].set_title(f"{bad_img}")
        axes[0].set_axis_off()
        for j in range(repeat):
            gen_img = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{metadata[bad_img]['gen_names'][j]}").convert("RGB")
            axes[1 + j].imshow(gen_img)
            axes[1 + j].set_title(f"{metadata[bad_img]['gen_names'][j]}")
            axes[1 + j].set_axis_off()
        fig.text(0.5, 0.04, f"{metadata[bad_img]['caption']}", ha='center')
        plt.tight_layout()
        plt.savefig(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root_bad}/{task_and_ov}/compare_{i}.png")
        plt.close(fig)

    return metadata


if __name__ == "__main__":
    # prompt = "<CAPTION>"
    # prompt = "<MORE_DETAILED_CAPTION>"
    # prompt = "<CAPTION_FIX>"
    prompt = "<CLASSES>"

    task = "10-10"
    replay_root = input("Enter replay root: ")
    overlap = input("Overlap? (y/n): ") == "y"
    repeat = int(input("Repeat images [recommended: 1-3]: "))
    print(f"Replay root: {replay_root}")
    print(f"Overlap: {overlap}")
    print(f"Repeat: {repeat}")
    seed = 19991
    img_gen_batch_size = 8

    classes = classes[:int(task.split("-")[0])]
    task_and_ov = f"{task}-ov" if overlap else task
    seed_everything(seed)
    for cl in classes:
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/{cl}/images", exist_ok=True)
        os.makedirs(f"/home/thesis/marx/wilson_gen/WILSON/{replay_root}/{task_and_ov}/{cl}/pseudolabels", exist_ok=True)
    config = OmegaConf.load("/home/thesis/marx/wilson_gen/stablediffusion/configs/stable-diffusion/v2-inference.yaml")


    image_names = get_image_names(task_and_ov)  # SLICE USING [:50] FOR DEBUGGING A MINIMAL FULL CHECK ON OVERLAPPED
    captions, metadata = get_captions(replay_root, task_and_ov, image_names, prompt, repeat, img_gen_batch_size)
    gen_images(replay_root, captions, metadata, repeat, img_gen_batch_size)
    bad_images = generate_labels(replay_root, task_and_ov, wilson_opts_obj(task, overlap))
    counter = 1
    while bad_images != []:
        if counter <= 3:
            print(f"Found {len(bad_images)} bad images, repeating bad images for the {counter}. time")
            metadata = repeat_bad_images(replay_root, task_and_ov, bad_images, prompt, repeat, img_gen_batch_size, metadata, seed, counter)
        else:
            print(f"Found {len(bad_images)} bad images, repeating bad images for the {counter}. time, reverting to base prompt")
            metadata = repeat_bad_images_base(replay_root, task_and_ov, bad_images, prompt, repeat, img_gen_batch_size, metadata, seed, counter)
        bad_images = generate_labels(replay_root, task_and_ov, wilson_opts_obj(task, overlap))
        counter += 1
    seed_everything(seed)
    control_replay_data(replay_root, task_and_ov, metadata)
    post_processing(replay_root, task_and_ov, metadata, repeat)

    print("Done!")