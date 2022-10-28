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
OUTPUT_ROOT="./results_2/"
OUTPUT=f"{OUTPUT_ROOT}data/"
os.makedirs(OUTPUT, exist_ok=True)

init_image = Image.open("ref.jpg").convert("RGB")
init_image = init_image.resize((512,512))

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	init_image=init_image,
	strength=0.75,
	guidance_scale=10,
	revision="fp16",
	torch_dtype=torch.float16,
	use_auth_token=True
).to("cuda")

count=0
##############################################
# run
##############################################
place_list=["town","wild","room","beach"]
breed_list=["American Shorthair","Scottish Fold","Ragdoll","British Shorthair","Russian Blue","Munchkin","Bengal Cat","Siamese Cat","Singapura","Osikat"]
color_list=["black","brown","gray","white"]
# pattern=["Solid color", "Stripes", "bi-color", "calico"]

metas = []
for iter in tqdm(range(generate_num)):
	
	place=random.choice(place_list)
	breed=random.choice(breed_list)
	color=random.choice(color_list)

	prompt = f"a photo of cat in the {place}. The breed is {breed}. the base color is {color}.The face is in a center."
	with autocast("cuda"):
		image = pipe(prompt)["sample"][0]  
	filename=f"{OUTPUT}cat_{iter:05}.png"
	image.save(filename)
	metas.append([filename, place[0], breed[2], color[0], prompt])
	print(prompt)
	print()

df_metas = pd.DataFrame(metas,
                  columns=['filename', 'place', 'breed',"color", "prompt"])

df_metas.to_csv(f"{OUTPUT_ROOT}metas.csv")

