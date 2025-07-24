#!/bin/bash

# Read the task string from the first argument
task="$1"
overlap="$2"
data_mode="$3"
k_inpainting="$4"
od_inpainting="$5"

# Split the task into n_init and n_incr
n_init=$(echo "$task" | cut -d'-' -f1)
n_incr=$(echo "$task" | cut -d'-' -f2)

# Calculate the number of steps
n_steps=$(( (20 - n_init) / n_incr ))

# Print the parsed values and set up configurations
echo "task: $task"
if [ "$overlap" = "overlap" ]; then
    echo "overlap: True"
    ov="-ov"
    gen_ov="y"
    ov_command="--overlap"
else
    echo "overlap: False"
    ov=""
    gen_ov="n"
    ov_command=""
fi

if [ "$data_mode" = "lora" ]; then
    echo "data_mode: lora"
    name="Incr_Gen_Lora"
else
    echo "data_mode: baseline"
    name="Incr_Gen_Base"
fi

if [ "$k_inpainting" = "k_inp" ]; then
    echo "k_inpainting: True"
    k_inp="--mask_pre_inp --inpainting_epoch 20 --inpainting_threshold 0.05"
    name="${name}_Inp_Thresh_05"
else
    echo "k_inpainting: False"
    k_inp="--mask_pre_inp"
fi

if [ "$od_inpainting" = "od_inp" ]; then
    echo "od_inpainting: True"
    od_inp="--inpainting_old_od"
    name="${name}_OD_Inp"
else
    echo "od_inpainting: False"
    od_inp=""
fi

name="${name}_RR"

port=$(python get_free_port.py)
echo ${port}
alias exp='torchrun --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=voc
epochs=40
lr=0.001
lr_init=0.01

path=checkpoints/step/${dataset}-${task}${ov}/
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 ${ov_command} --val_interval 2"
pretr=${path}Base_0.pth

replay_root=replay_data_multistep/${name}

# Loop from 0 to n_steps (inclusive)
for (( i=1; i<=n_steps; i++ ))
do
    echo "Generating step $(( i-1 ))"
    if [ $i -eq 1 ]; then
        if [ "$data_mode" = "lora" ]; then
            name_command="Base"
            replay_root_command=replay_data_multistep/Lora_Base
        else
            name_command="Base"
            replay_root_command=replay_data_multistep/Base_Base
        fi
    else
        name_command=${name}
        replay_root_command=${replay_root}
    fi
    PYTHONPATH=.. /home/thesis/miniconda3/envs/hugface12/bin/python  /home/thesis/marx/wilson_gen/scripts/make_dataset_multistep.py ${replay_root_command} ${task} ${gen_ov} $(( i-1 )) 1 ${name_command}_$(( i-1 ))
    echo "Training step $i"
    exp --name ${name} --step ${i} --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs} --replay --replay_root ${replay_root_command} --replay_size 1 ${k_inp} ${od_inp}
    pretr=${path}${name}_${i}.pth
done
