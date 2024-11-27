from segmentation_module import make_model
import pickle
import tasks
import os
import torch
import numpy as np
from PIL import Image
from dataset import transform
from tqdm import tqdm

_transform = transform.Compose([
    # transform.Resize(size=512),  # Not necessary, gen_imgs are 512x512 already
    # transform.CenterCrop(size=512),
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

class_to_idx = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "dining_table": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "potted_plant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tv_monitor": 20
}

def get_pseudo_labeler(opts):
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    checkpoint = torch.load(opts.ckpt, map_location="cpu", weights_only=False)
    net_dict = model.state_dict()
    pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items() if
        (k.replace("module.", "") in net_dict) and
        (v.shape == net_dict[k.replace("module.", "")].shape)}
    assert pretrained_dict.keys() == net_dict.keys(), "STATE DICTS NOT COMPATIBLE!"
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    del checkpoint
    model.to("cuda")
    for par in model.parameters():
            par.requires_grad = False
    model.eval()
    return model

def generate_pseudo_labels(replay_path_w_task, opts):
    pseudo_labeler = get_pseudo_labeler(opts)
    class_folders = os.listdir(replay_path_w_task)
    for cf in tqdm(class_folders, leave=True):
        # if cf not in list(class_to_idx.keys()):
        #      print("Skipping class", cf)
        #      continue
        pseudolabels_1h = dict()
        img_names = os.listdir(os.path.join(replay_path_w_task, cf, "images"))
        for img_name in tqdm(img_names, leave=False):
            img = Image.open(os.path.join(replay_path_w_task, cf, "images", img_name)).convert("RGB")
            img = _transform(img).to("cuda", dtype=torch.float32).unsqueeze(0)
            outputs = pseudo_labeler(img)[0].cpu().numpy().squeeze()
            pseudo_label = np.argmax(outputs, axis=0).astype(np.uint8)

            pseudo_label_1h = np.zeros((21), dtype=np.uint8)
            for present_class in np.unique(pseudo_label):
                    pseudo_label_1h[int(present_class)] = 1
            pseudo_label_1h[class_to_idx[cf]] = 1
            pseudolabels_1h[img_name] = pseudo_label_1h[1:] # Exclude background class

            pseudo_label = Image.fromarray(pseudo_label)
            pseudo_label.save(os.path.join(replay_path_w_task, cf, "pseudolabels", img_name))
        with open(os.path.join(replay_path_w_task, cf, "pseudolabels_1h.pkl"), "wb") as f:
            pickle.dump(pseudolabels_1h, f)


if __name__ == "__main__":
    import os
    os.chdir("/home/thesis/marx/wilson_gen/WILSON_WIP")
    class opts_obj():
        def __init__(self):
            pass
    opts = opts_obj()
    opts.dataset = "voc"
    opts.task = "10-10"
    opts.step = 0
    opts.test = True
    opts.ckpt = "./checkpoints/step/voc-10-10-ov/Base_0.pth"
    opts.norm_act = "iabn_sync"
    opts.backbone = "resnet101"
    opts.output_stride = 16
    opts.no_pretrained = False
    opts.pooling = 32
    replay_path = f"./replay_data/{opts.task}/"
    generate_pseudo_labels(replay_path, opts)
    print("Successfully generated pseudolabels!")