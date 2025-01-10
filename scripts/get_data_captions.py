import os
os.chdir(os.path.dirname(__file__))
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM


def get_image_names():
    idxs_path = f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{task_and_ov}/train-0.npy"
    idxs = np.load(idxs_path)
    # print(len(idxs))
    # print(idxs)
    with open("/home/thesis/marx/wilson_gen/WILSON/data/voc/splits/train_aug.txt") as f:
        lines = f.readlines()
        lines = [l.strip().split(" ")[0][1:] for l in lines]
        image_names = np.array(lines)[idxs]
    return image_names


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    return model, processor


def save_captions(model, processor, image_names):
    prompt = "<MORE_DETAILED_CAPTION>"
    captions = []
    for image_name in tqdm(image_names):
        image = Image.open(f"/home/thesis/marx/wilson_gen/WILSON/data/voc/{image_name}").convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        captions.append(parsed_answer[prompt])
    with open(f"/home/thesis/marx/wilson_gen/voc_captions_{task_and_ov}.txt", "w") as f:
        for caption in captions:
            caption = caption.replace("\n", " ")
            f.write(caption + "\n")

def main():
    image_names = get_image_names()
    model, processor = load_model()
    save_captions(model, processor, image_names)
    print("Done!")

if __name__ == "__main__":

    task = "10-10"
    overlap = True

    device = "cuda"
    torch_dtype = torch.float32
    model_id = "microsoft/Florence-2-large"

    ov_string = "-ov" if overlap else ""
    task_and_ov = f"{task}{ov_string}"

    main()