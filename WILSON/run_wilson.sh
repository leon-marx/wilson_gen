#!/bin/bash

# Run base model
torchrun --nproc_per_node=1 --master_port 29500 run.py --num_workers 12 --name Base --step 0 --lr 0.01 --bce --dataset voc --task 10-10 --batch_size 24 --epochs 30 --val_interval 2

# Run incremental step
torchrun --nproc_per_node=1 --master_port 29500 run.py --num_workers 12 --name Incr --step 1 --weakly --lr 0.001 --alpha 0.5 --step_ckpt checkpoints/step/voc-10-10/Base_o.pth --loss_de 1 --lr_policy warmup --affinity --dataset voc --task 10-10 --batch_size 24 --epochs 40