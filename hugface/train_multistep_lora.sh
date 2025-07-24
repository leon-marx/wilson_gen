#!/bin/bash

# this script trains loras for the multistep tasks (e.g. 10-1)
# so far, a new lora is trained from scratch for each step
# this is not optimal, but it is the most simple starting point
# call like ./train_multistep_lora.sh 10-10 overlap

alias exp='accelerate launch --config_file base_config.yaml train_text_to_image_lora.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --validation_prompt="photo of a CLASS" --seed=12321 --center_crop --random_flip --train_batch_size=8 --num_train_epochs=40 --gradient_checkpointing --lr_scheduler="cosine" --lr_warmup_steps=0 --dataloader_num_workers=4 --max_grad_norm=1.0 --mixed_precision="no" --report_to="wandb" --enable_xformers_memory_efficient_attention --gradient_accumulation_steps=8 --num_validation_images=40 --rank=128 --learning_rate=1e-03 --checkpointing_epochs=1 --checkpointing_steps=1000000'
shopt -s expand_aliases

if [ $2 = overlap ]; then
    # echo "Overlap"
    ov="-ov"
else
    # echo "Disjoint"
    ov=""
fi

# echo "Training lora for multistep task $1$ov"

if [ $1 = 10-1 ] || [ $1 = all ]; then
    for step in 0 1 2 3 4 5 6 7 8 9 10; do
        # echo "Training step $step"
        # echo multistep_voc_lora_top_100_10-1$ov/$step
        # echo best_lora_checkpoints/10-1$ov/$step
        exp --dataset_name="multistep_voc_lora_top_100_10-1$ov/$step" --output_dir="best_lora_checkpoints/10-1$ov/$step"
    done
fi

if [ $1 = 10-5 ] || [ $1 = all ]; then
    for step in 0 1 2; do
        # echo "Training step $step"
        # echo multistep_voc_lora_top_100_10-5$ov/$step
        # echo best_lora_checkpoints/10-5$ov/$step
        exp --dataset_name="multistep_voc_lora_top_100_10-5$ov/$step" --output_dir="best_lora_checkpoints/10-5$ov/$step"
    done
fi

if [ $1 = 10-10 ] || [ $1 = all ]; then
    for step in 0 1; do
        # echo "Training step $step"
        # echo multistep_voc_lora_top_100_10-10$ov/$step
        # echo best_lora_checkpoints/10-10$ov/$step
        exp --dataset_name="multistep_voc_lora_top_100_10-10$ov/$step" --output_dir="best_lora_checkpoints/10-10$ov/$step"
    done
fi

if [ $1 = 15-1 ] || [ $1 = all ]; then
    for step in 0 1 2 3 4 5; do
        # echo "Training step $step"
        # echo multistep_voc_lora_top_100_15-1$ov/$step
        # echo best_lora_checkpoints/15-1$ov/$step
        exp --dataset_name="multistep_voc_lora_top_100_15-1$ov/$step" --output_dir="best_lora_checkpoints/15-1$ov/$step"
    done
fi

if [ $1 = 15-5 ] || [ $1 = all ]; then
    for step in 0 1; do
        # echo "Training step $step"
        # echo multistep_voc_lora_top_100_15-5$ov/$step
        # echo best_lora_checkpoints/15-5$ov/$step
        exp --dataset_name="multistep_voc_lora_top_100_15-5$ov/$step" --output_dir="best_lora_checkpoints/15-5$ov/$step"
    done
fi
