from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
import time

print("Loading models...")
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
base.to("cuda")
base.set_progress_bar_config(disable=True)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
refiner.set_progress_bar_config(disable=True)

n_steps = 40
high_noise_frac = 0.8
prompts = [
    "realistic photo of a person in its typical environment",
    "realistic photo of a bird in its typical environment",
    "realistic photo of a cat in its typical environment",
    "realistic photo of a cow in its typical environment",
    "realistic photo of a dog in its typical environment",
    "realistic photo of a horse in its typical environment",
    "realistic photo of a sheep in its typical environment",
    "realistic photo of a aeroplane in its typical environment",
    "realistic photo of a bicycle in its typical environment",
    "realistic photo of a boat in its typical environment",
    "realistic photo of a bus in its typical environment",
    "realistic photo of a car in its typical environment",
    "realistic photo of a motorbike in its typical environment",
    "realistic photo of a train in its typical environment",
    "realistic photo of a bottle in its typical environment",
    "realistic photo of a chair in its typical environment",
    "realistic photo of a dining table in its typical environment",
    "realistic photo of a potted plant in its typical environment",
    "realistic photo of a sofa in its typical environment",
    "realistic photo of a tv/monitor in its typical environment",
]

print("Generating images...")
time.sleep(1)
for prompt in tqdm(prompts, leave=True):
    for i in tqdm(range(100), leave=False):
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        image.save(f"gen_imgs/{prompt.replace('/', '_')}_{i}.jpg")