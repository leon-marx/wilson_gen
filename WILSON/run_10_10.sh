#!/bin/bash

# These are the base WILSON-like runs for the 10-1 setting, both disjoint and overlap.

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=40
task=10-10
lr=0.001
lr_init=0.01



if [ $1 = disjoint ] || [ $1 = both ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
  echo "Disjoint"
  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  exp --name Base --step 0 --bce --lr ${lr_init} ${dataset_pars}  --epochs 30
  pretr=${path}Base_0.pth
  exp --name Incr --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs}
  exp --name Incr_OD_Inp --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --inpainting_old_od

fi

if [ $1 = overlap ] || [ $1 = both ]; then
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  exp --name Base --step 0 --bce --lr ${lr_init} ${dataset_pars}  --epochs 30
  pretr=${path}Base_0.pth
  exp --name Incr --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs}
  exp --name Incr_OD_Inp --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --inpainting_old_od

fi
