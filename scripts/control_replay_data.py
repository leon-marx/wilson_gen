# Sometimes, stablediffusion generates corrupted images.
# This file checks, that all images can be loaded.

import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

os.chdir("/home/thesis/marx/wilson_gen/WILSON/replay_data/10-10")
for dir in os.listdir():
     for img in os.listdir(f"{dir}/images"):
         imgg = Image.open(f"{dir}/images/{img}")
         try:
             print(pil_to_tensor(imgg).shape)
         except OSError:
             print(dir, img)
             raise TypeError