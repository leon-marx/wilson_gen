#!/bin/bash

port=$(python get_free_port.py)
echo ${port}

# Debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Maybe not necessary
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = ${port}

# Run base model
torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --name Base --step 0 --lr 0.01 --bce --dataset voc --task 10-10 --batch_size 16 --epochs 30 --val_interval 2

# Run incremental step
torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --name Incr --step 1 --weakly --lr 0.001 --alpha 0.5 --step_ckpt checkpoints/step/voc-10-10/Base_0.pth --loss_de 1 --lr_policy warmup --affinity --dataset voc --task 10-10 --batch_size 16 --epochs 40