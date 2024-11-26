from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.utils.data.dataset import ConcatDataset
import math
import os

import torch.multiprocessing as mp

class InterleaveSampler(Sampler):
    def __init__(self, dataset, batch_size, max_iterations = None, shuffle=False):
        self.dataset = dataset
        self.datasets_num = len(self.dataset.datasets)
        self.batch_size = batch_size
        self.max_iter = max_iterations
        self.shuffle = shuffle
        self.largest_dataset_size = max([len(d) for d in self.dataset.datasets])
        self.minimum_dataset_size = min([len(d) for d in self.dataset.datasets])
        # print()

    def __iter__(self):
        # print("sampler")
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.datasets_num):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if self.shuffle:
                sampler = RandomSampler(cur_dataset)
            else:
                sampler = SequentialSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.datasets_num // 2
        samples_to_grab = self.batch_size//2
        if not self.max_iter is None:
            epoch_samples = self.max_iter*self.batch_size
        else:
            epoch_samples = self.largest_dataset_size * self.datasets_num

        final_sample_list = []
        for _ in range(0,epoch_samples, step):
            for i in range(self.datasets_num):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for __ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # print(i)
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_sample_list.extend(cur_samples)

        return iter(final_sample_list)

    def __len__(self):
        if not self.max_iter is None:
            return self.max_iter*self.batch_size
        else:
            return math.ceil(self.largest_dataset_size/self.batch_size) * self.datasets_num * self.batch_size



class DistributedInterleaveSampler(Sampler):
    def __init__(self, sampler, num_replicas, rank=None, shuffle=False):
        self.sampler = sampler
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(sampler.__len__() * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(self.sampler)
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
