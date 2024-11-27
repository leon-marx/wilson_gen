import torch
import random

class InterleaveSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.replay_ratio = self.dataset.replay_ratio
        self.num_voc = self.dataset.num_voc
        self.num_gen = len(self.dataset) - self.num_voc
        self.gen_per_batch = int(self.batch_size * self.replay_ratio)
        self.voc_per_batch = int(self.batch_size - self.gen_per_batch)
        self.num_batches = self.num_voc // self.voc_per_batch
        assert self.num_gen // self.gen_per_batch >= self.num_voc // self.voc_per_batch, "Not enough gen data, create more!"

    def __iter__(self):
        voc_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        voc_generator = torch.Generator()
        voc_generator.manual_seed(voc_seed)
        gen_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        gen_generator = torch.Generator()
        gen_generator.manual_seed(gen_seed)
        voc_inds = torch.randperm(self.num_voc, generator=voc_generator).tolist()
        gen_inds = (torch.randperm(self.num_gen, generator=gen_generator) + self.num_voc).tolist()
        full_inds = []
        for i in range(self.num_batches):
            batch = voc_inds[i * self.voc_per_batch:(i+1) * self.voc_per_batch]
            batch += gen_inds[i * self.gen_per_batch:(i+1) * self.gen_per_batch]
            random.shuffle(batch)
            full_inds += batch
        yield from full_inds

    def __len__(self):
        return self.batch_size * self.num_batches

    def set_epoch(self, epoch):
        pass

# DEBUGGING
# if __name__ == "__main__":
#     import os
#     os.chdir("/home/thesis/marx/wilson_gen/WILSON_WIP")
#     from dataset import VOCGenSegmentationIncremental
#     import tasks
#     from dataset import transform
#     step_dict = tasks.get_task_dict("voc", "10-10", 1)
#     train_transform = transform.Compose([
#         transform.RandomResizedCrop(512, (0.5, 2)),
#         transform.RandomHorizontalFlip(),
#         transform.ToTensor(),
#         transform.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#     ])
#     my_dset = VOCGenSegmentationIncremental(
#         root="data",
#         replay_root="./replay_data",
#         replay_ratio=0.4,
#         task="10-10",
#         step_dict=step_dict,
#         train=True,
#         transform=train_transform,
#         idxs_path="data/voc/10-10/train-1.npy",
#         masking_value=0,
#         masking=True,
#         overlap=True,
#         step=1,
#         weakly=True,
#         pseudo=None
#     )
#     my_sampler = InterleaveSampler(my_dset, 24)

#     train_loader = torch.utils.data.DataLoader(my_dset, batch_size=24,
#                                     sampler=InterleaveSampler(my_dset, 24),
#                                     num_workers=1, drop_last=True)
#     # import matplotlib.pyplot as plt
#     from tqdm import tqdm
#     for batch in tqdm(train_loader):
#         # fig, axes = plt.subplots(4, 6, figsize=(24, 16))
#         # for j, img in enumerate(batch[0]):
#         #     axes[j//6,j%6].imshow(img.permute(1, 2, 0).numpy())
#         #     axes[j//6,j%6].set_xticks([])
#         #     axes[j//6,j%6].set_yticks([])
#         # plt.tight_layout()
#         # plt.savefig(f"sampler_testttt_{i}.png")
#         inds = batch[2].numpy().flatten()
#         n_voc = np.sum(inds < train_loader.sampler.num_voc)
#         n_gen = np.sum(inds >= train_loader.sampler.num_voc)
#         if n_voc != 15 or n_gen != 9:
#             print("OH NO")
#         # print(f"{n_voc = }")
#         # print(f"{n_gen = }")
#         # if i >= 1:
#         #     break
#     # for i in my_sampler:
#     #     print(i)
#     #     break