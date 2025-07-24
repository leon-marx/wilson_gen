#!/bin/bash

# Improved inpainting using thresholding and masking background in replay images without new classes
# disjoint block contains additional lines concerning the current sota in overlapped to bring it up to date
# (for overlapped, these lines are scattered in the other run_XY.sh files)

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=40
task=10-10
lr=0.001


if [ $1 = disjoint ] || [ $1 = both ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
  echo "Disjoint"
  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  pretr=${path}Base_0.pth

  if [ $2 = baseline ] || [ $2 = both ]; then
    # exp --name Incr --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs}
    exp --name Incr_Gen_Base_Rep_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1
    # exp --name Incr_Gen_Base_Pre_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp
    # exp --name Incr_Gen_Base_Inp_Thresh_05_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.05
    # exp --name Incr_Gen_Base_Inp_Thresh_20_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.2
    # exp --name Incr_Gen_Base_Inp_Thresh_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.1
  fi

  if [ $2 = lora ] || [ $2 = both ]; then
    exp --name Incr_Gen_ODInp_pure --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --inpainting_old_od
    exp --name Incr_Gen_Lora_Rep_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1
    exp --name Incr_Gen_Lora_Pre_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp
    exp --name Incr_Gen_Lora_Inp_Thresh_05_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.05
    exp --name Incr_Gen_Lora_Inp_Thresh_20_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.2
    exp --name Incr_Gen_Lora_Inp_Thresh_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.1
  fi
fi

if [ $1 = overlap ] || [ $1 = both ]; then
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  pretr=${path}Base_0.pth

  if [ $2 = baseline ] || [ $2 = both ]; then
    # exp --name Incr_Gen_Base_Inp_Thresh_05_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.05
    exp --name Incr_Gen_Base_Inp_Thresh_20_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.2
    # exp --name Incr_Gen_Base_Inp_Thresh_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.1
    # exp --name Incr_Gen_Base_Inp_Thresh_Post_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --mask_post_inp --inpainting_threshold 0.1
  fi

  if [ $2 = lora ] || [ $2 = both ]; then
    exp --name Incr_Gen_Lora_Inp_Thresh_05_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.05
    exp --name Incr_Gen_Lora_Inp_Thresh_20_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.2
    # exp --name Incr_Gen_Lora_Inp_Thresh_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.1
    # exp --name Incr_Gen_Lora_Inp_Thresh_Post_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --mask_pre_inp --inpainting_epoch 20 --mask_post_inp --inpainting_threshold 0.1
  fi
fi