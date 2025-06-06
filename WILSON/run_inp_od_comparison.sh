#!/bin/bash

# Run several experiments using good background masking and simple inpainting

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=40
task=10-10
lr=0.001


path=checkpoints/step/${dataset}-${task}-ov/
ov="--overlap"
echo "Overlap"
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
pretr=${path}Base_0.pth

if [ $1 = od ]; then
  ckpt_path=${path}Incr_Gen_Base_Pre_Inp_RR_1_epoch_20.pth
  # exp --name Incr_Gen_Base_Rep_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --ckpt_interval 10
  # exp --name Incr_Gen_ODInp_both --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --inpainting_old_od --inpainting_old
  # exp --name Incr_Gen_Base_Pre_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --ckpt_interval 10 --mask_pre_inp
  # exp --name Incr_Gen_Base_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --ckpt $ckpt_path --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --ckpt_interval 10 --mask_pre_inp --inpainting_epoch 20
  exp --name Incr_Gen_Base_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_baseline --replay_size 1 --ckpt_interval 10 --mask_pre_inp --inpainting_epoch 20

elif [ $1 = both ]; then
  ckpt_path=${path}Incr_Gen_Lora_Pre_Inp_RR_1_epoch_20.pth
  # exp --name Incr_Gen_Lora_Rep_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --ckpt_interval 10
  # exp --name Incr_Gen_ODInp_pure --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --inpainting_old_od
  # exp --name Incr_Gen_Lora_Pre_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --ckpt_interval 10 --mask_pre_inp
  # exp --name Incr_Gen_Lora_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --ckpt $ckpt_path --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --ckpt_interval 10 --mask_pre_inp --inpainting_epoch 20
  exp --name Incr_Gen_Lora_Inp_RR --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root replay_data_lora --replay_size 1 --ckpt_interval 10 --mask_pre_inp --inpainting_epoch 20

fi