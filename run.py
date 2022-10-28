#https://zenn.dev/chili/articles/ba5aa2057498f9
#ref https://mononoco.com/creative/tool/stable-diffusion/windows-in-docker
#
#git config --global credential.helper store
#huggingface-cli login

##############################################
# make sure you're logged in with `huggingface-cli login`
##############################################
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline

import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random

generate_num=1000

##############################################
# init
##############################################
OUTPUT_ROOT="./results_single/"
OUTPUT=f"{OUTPUT_ROOT}data/"
os.makedirs(OUTPUT, exist_ok=True)

init_image = Image.open("ref.jpg").convert("RGB")
init_image = init_image.resize((512,512))

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	revision="fp16",
	torch_dtype=torch.float16,
	use_auth_token=True
).to("cuda")

count=0
##############################################
# run
##############################################

prompt = f"a illustraion of chainsaw man."
with autocast("cuda"):
	image = pipe(prompt)["sample"][0]  
filename=f"{OUTPUT}_{prompt}.png"
image.save(filename)