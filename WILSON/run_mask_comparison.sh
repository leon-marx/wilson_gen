#!/bin/bash

# This script was used to run the exhaustive masking and inpainting sweeps. Masking is naive here, results are not good.

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=20
task=10-10
lr=0.001


path=checkpoints/step/${dataset}-${task}-ov/
ov="--overlap"
echo "Overlap"
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 1"
pretr=${path}Base_0.pth

if [ $1 = baseline ]; then
  replay_root=replay_data_baseline
  # exp --name Incr_Gen_Base_Inpainting_Stub --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 10 --replay --replay_root ${replay_root} --replay_size 1 --inpainting
  # stub_ckpt=${path}Incr_Gen_Base_Inpainting_Stub_1.pth
  # exp --name Incr_Gen_Base_New1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_lde_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_lde --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_l_seg_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_seg --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_l_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_l_cls_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cls --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_l_cam_new_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cam_new --inpainting
  # exp --name Incr_Gen_Base_New1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_lde_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_lde --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_l_seg_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_seg --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_l_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_l_cls_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cls --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_l_cam_new_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cam_new --inpainting_old --inpainting
  # exp --name Incr_Gen_Base_Both1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old --inpainting

  # exp --name Incr_Gen_Base_No1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  # exp --name Incr_Gen_Base_No1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new
  # exp --name Incr_Gen_Base_Old1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting_old
  # exp --name Incr_Gen_Base_Old1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old

  # exp --name Incr_Gen_Base_Both1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 40 --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old --inpainting

  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  exp --name Incr_Gen_Base_Old1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 40 --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old

elif [ $1 = lora ]; then
  replay_root=replay_data_lora
  # exp --name Incr_Gen_Lora_Inpainting_Stub --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 10 --replay --replay_root ${replay_root} --replay_size 1 --inpainting
  # stub_ckpt=${path}Incr_Gen_Lora_Inpainting_Stub_1.pth
  # exp --name Incr_Gen_Lora_New1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_lde_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_lde --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_l_seg_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_seg --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_l_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_l_cls_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cls --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_l_cam_new_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cam_new --inpainting
  # exp --name Incr_Gen_Lora_New1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_lde_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_lde --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_l_seg_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_seg --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_l_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_l_cls_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cls --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_l_cam_new_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_cam_new --inpainting_old --inpainting
  # exp --name Incr_Gen_Lora_Both1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old --inpainting

  # exp --name Incr_Gen_Lora_No1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1
  # exp --name Incr_Gen_Lora_No1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new
  # exp --name Incr_Gen_Lora_Old1h_Masked_none_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --inpainting_old
  # exp --name Incr_Gen_Lora_Old1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old

  # exp --name Incr_Gen_Lora_Both1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 40 --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old --inpainting

  dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
  exp --name Incr_Gen_Lora_Old1h_Masked_all_loc_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs 40 --replay --replay_root ${replay_root} --replay_size 1 --mask_replay_l_loc --mask_replay_l_cls --mask_replay_l_cam_new --inpainting_old

fi