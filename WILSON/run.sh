#!/bin/bash

# We assume to have a parameter indicating whether to use overlap (0 or 1)
port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

overlap=1  # set to 1 to activate
dataset=voc
epochs=40
task=10-10
lr_init=0.01
lr=0.001

if [ ${overlap} -eq 0 ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
else
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
fi

dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"

exp --name Base --step 0 --bce --lr ${lr_init} ${dataset_pars}  --epochs 30

pretr=${path}Base_0.pth

exp --name Incr --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs}
