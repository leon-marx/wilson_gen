#!/bin/bash

# for seed in 211 212 213 214 215 216 217 218 219 220
# do
#     echo ${seed}
#     PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --outdir gen_imgs_baseline/ --n_iter 100 --n_samples 10 --repeat 1 --from-file ../voc_baseline_prompts.txt --ddim_eta 0.5 --n_rows 0 --seed ${seed}
# done

PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --outdir gen_base_refill --n_iter 1 --n_samples 8 --repeat 1 --from-file ../voc_baseline_prompts_refill_rest.txt --ddim_eta 0.5 --n_rows 0 --seed 99

# PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --outdir gen_mrte_refill --n_iter 1 --n_samples 1 --repeat 1 --from-file ../voc_mrte_prompts_refill.txt --ddim_eta 0.5 --n_rows 0 --seed 99