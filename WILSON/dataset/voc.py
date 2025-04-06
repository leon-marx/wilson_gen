import os
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset
import numpy as np
import pickle
from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}

coco_map = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if as_coco:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug_ascoco.txt')
            else:
                split_f = os.path.join(splits_dir, 'val_ascoco.txt')
        else:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        if pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
            else:
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"PseudoLabels/{pseudo}/rw")) for x in
                               self.images]
        if as_coco:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"cocovoc_1h_labels_{self.image_set}.npy"))
        else:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))

        self.indices = indices if indices is not None else np.arange(len(self.images))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_lvl_lbls

    def __len__(self):
        return len(self.indices)

class VOCGenSegmentation(data.Dataset):
    def __init__(self,
                 root,
                 replay_root,
                 replay_ratio,
                 replay_size,
                 task,
                 overlap,
                 train=True,
                 transform=None,
                 indices=None,
                 saliency=False,
                 pseudo=None,):
        assert train, "Replay only works for training"
        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        split_f = os.path.join(splits_dir, 'train_aug.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]

        if pseudo is not None and train:
            self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
        self.img_lvl_labels = [one_hot for one_hot in np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))]

        self.indices = indices if indices is not None else np.arange(len(self.images))
        self.indices = [idx for idx in self.indices]

        # Replay
        ov_string = "-ov" if overlap else ""
        if task == "10-10":
            old_classes = [classes[i] for i in range(1, 11, 1)]
        elif task == "15-5":
            old_classes = [classes[i] for i in range(1, 16, 1)]
        self.num_voc = len(self.indices)
        self.replay_root = replay_root
        self.replay_images = []
        self.replay_1h_lbls = []
        with open(os.path.join(replay_root, f"{task}{ov_string}", "class_counts.pkl"), "rb") as f:
                self.class_counts = pickle.load(f)
        self.old_classes = old_classes
        self.task = task
        self.ov_string = ov_string
        self.img_names = {}
        for old_class in old_classes:
            if replay_size is not None:
                # print(F"{replay_size * self.class_counts[old_class] = }")
                # print("Original number of data:", len(sorted(os.listdir(os.path.join(replay_root, f"{task}{ov_string}", old_class, "images")))))
                img_names = sorted(os.listdir(os.path.join(replay_root, f"{task}{ov_string}", old_class, "images")))[:replay_size * self.class_counts[old_class]]
                # print("New number of data:", len(img_names))
            else:
                img_names = sorted(os.listdir(os.path.join(replay_root, f"{task}{ov_string}", old_class, "images")))
            self.img_names[old_class] = img_names
            self.replay_images += [(os.path.join(replay_root, f"{task}{ov_string}", old_class, "images", img_name),
                               os.path.join(replay_root, f"{task}{ov_string}", old_class, "pseudolabels", img_name[:-4] + ".png")) for img_name in img_names]
            with open(os.path.join(replay_root, f"{task}{ov_string}", old_class, "pseudolabels_1h.pkl"), "rb") as f:
                pseudolabels_1h = pickle.load(f)
            self.replay_1h_lbls += [pseudolabels_1h[img_name[:-4] + ".png"] for img_name in img_names]

    def update_pseudolabels(self):
        self.replay_1h_lbls = []
        for old_class in self.old_classes:
            with open(os.path.join(self.replay_root, f"{self.task}{self.ov_string}", old_class, f"inpainted_pseudolabels_1h.pkl"), "rb") as f:
                pseudolabels_1h = pickle.load(f)
            self.replay_1h_lbls += [pseudolabels_1h[img_name[:-4] + ".png"] for img_name in self.img_names[old_class]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if index < self.num_voc:
            img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
            target = Image.open(self.images[self.indices[index]][1])
            img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

            if self.transform is not None:
                img, target = self.transform(img, target)
            return img, target, img_lvl_lbls
        else:
            img = Image.open(self.replay_images[index - self.num_voc][0]).convert('RGB')
            target = Image.open(self.replay_images[index - self.num_voc][1])
            img_lvl_lbls = self.replay_1h_lbls[index - self.num_voc]

            if self.transform is not None:
                img, target = self.transform(img, target)
            return img, target, img_lvl_lbls
    def __len__(self):
        return len(self.indices) + len(self.replay_images)

class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        return full_voc

class VOCGenSegmentationIncremental(IncrementalSegmentationDataset):
    def __init__(self,
                 root,
                 replay_root,
                 replay_ratio,
                 replay_size,
                 step_dict,
                 task,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=0,
                 step=0,
                 weakly=False,
                 pseudo=None,):
        self.replay_root = replay_root
        self.replay_ratio = replay_ratio
        self.replay_size = replay_size
        self.task = task
        self.overlap = overlap
        super().__init__(root=root,
            step_dict=step_dict,
            train=train,
            transform=transform,
            idxs_path=idxs_path,
            masking=masking,
            overlap=overlap,
            masking_value=masking_value,
            step=step,
            weakly=weakly,
            pseudo=pseudo)

    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCGenSegmentation(root=root, replay_root=self.replay_root, replay_ratio=self.replay_ratio, replay_size=self.replay_size, task=self.task, overlap=self.overlap, train=train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        self.full_voc = full_voc
        self.num_voc = self.full_voc.num_voc
        return self.full_voc

    def update_pseudolabels(self):
        self.full_voc.update_pseudolabels()

class VOCasCOCOSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, as_coco=True,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc

class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])
