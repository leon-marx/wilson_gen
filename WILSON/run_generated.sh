#!/bin/bash

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=40
task=10-10
lr=0.001

if [ $1 = baseline ]; then
  echo "Running baseline"
  replay_root=replay_data_baseline
  # Baseline disjoint
  if [ $2 = disjoint ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}/
    ov=""
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_Base_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  fi

  # Baseline overlap
  if [ $2 = overlap ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}-ov/
    ov="--overlap"
    echo "Overlap"
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_Base_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  fi

elif [ $1 = mrte ]; then
  echo "Running MRTE"
  replay_root=replay_data_mrte
  # MRTE disjoint
  if [ $2 = disjoint ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}/
    ov=""
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_MRTE_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  fi

  # MRTE overlap
  if [ $2 = overlap ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}-ov/
    ov="--overlap"
    echo "Overlap"
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_MRTE_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  fi

elif [ $1 = captions ]; then
  echo "Running captions"
  replay_root=replay_data_better_cap
  # captions disjoint
  if [ $2 = disjoint ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}/
    ov=""
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_Better_Cap_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
    exp --name Incr_Gen_Better_Cap_Large_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_ratio 0.5
  fi

  # captions overlap
  if [ $2 = overlap ] || [ $2 = all ]; then
    path=checkpoints/step/${dataset}-${task}-ov/
    ov="--overlap"
    echo "Overlap"
    dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
    pretr=${path}Base_0.pth
    exp --name Incr_Gen_Better_Cap_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
    exp --name Incr_Gen_Better_Cap_Large_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_ratio 0.5
  fi

else
  echo "Invalid argument provided, ./run_generated.sh [baseline/mrte/captions] [overlap/disjoint/all]"
fi