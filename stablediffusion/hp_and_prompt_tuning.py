from subprocess import Popen
import os
import time

os.chdir("/home/thesis/marx/wilson_gen/stablediffusion")

steps_list = [10, 30, 50, 100]
sampler_list = ["ddim", "plms", "dpm"]
ddim_eta_list = [0.0, 0.2, 0.4, 0.8, 1.0]
scale_list = [0.0, 5.0, 10.0, 15.0, 20.0]

# outdir = "gen_imgs/"

prompt = "realistic and modern photo of a aeroplane in its typical environment"

for sampler in sampler_list:
    if sampler == "ddim":
        for ddim_eta in ddim_eta_list:
            for scale in scale_list:
                for steps in steps_list:
                    outdir = f"hp_search/{sampler}/eta_{ddim_eta}/scale_{scale}/steps_{steps}".replace(".", "_")
                    print(outdir)
                    start = time.time()
                    Popen(f"PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --seed 171717 --outdir {outdir} --n_iter 4 --n_samples 16 --prompt '{prompt}' --ddim_eta {ddim_eta} --steps {steps} --scale {scale}", shell=True).wait()
                    end = time.time()
                    print(end - start)
                    with open(f"{outdir}/time.txt", "w") as f:
                        f.write(str(end - start))
    else:
        for scale in scale_list:
            for steps in steps_list:
                print(outdir)
                start = time.time()
                outdir = f"hp_search/{sampler}/scale_{scale}/steps_{steps}".replace(".", "_")
                Popen(f"PYTHONPATH=. python scripts/txt2img.py --ckpt pretrained/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --device cuda --seed 171717 --outdir {outdir} --n_iter 4 --n_samples 16 --prompt '{prompt}' --{sampler} --steps {steps} --scale {scale}", shell=True).wait()
                end = time.time()
                print(end - start)
                with open(f"{outdir}/time.txt", "w") as f:
                    f.write(str(end - start))

print("----------------------------------------DONE----------------------------------------")

for sampler in sampler_list:
    if sampler == "ddim":
        for ddim_eta in ddim_eta_list:
            for scale in scale_list:
                for steps in steps_list:
                    outdir = f"hp_search/{sampler}/eta_{ddim_eta}/scale_{scale}/steps_{steps}".replace(".", "_")
                    print(outdir)
                    with open(f"{outdir}/time.txt", "r") as f:
                        print("Time taken for 4x16 images:", f.read(), "seconds")
                        print("")
    else:
        for scale in scale_list:
            for steps in steps_list:
                print(outdir)
                with open(f"{outdir}/time.txt", "r") as f:
                    print("Time taken for 4x16 images:", f.read(), "seconds")
                    print("")
