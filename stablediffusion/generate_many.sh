#!/bin/bash

for seed in 101 102 103 104 105 106 107 108 109
do
    echo ${seed}
    PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --outdir gen_imgs/ --n_iter 1 --n_samples 1 --repeat 50 --from-file prompts.txt --ddim_eta 0.5 --n_rows 0 --seed ${seed}
done