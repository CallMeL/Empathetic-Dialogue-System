# https://huggingface.co/HannahLin271/nanoGPT_single_conversation/resolve/main/pytorch_model.bin
import os
import torch
from nanoGPT import GPTConfig, GPT
from huggingface_hub import hf_hub_download
import shutil
import re
import sys
out_dir = "./out"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import requests
from pathlib import Path
from tqdm import tqdm
import gradio as gr

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status() 

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 

    # Create a progress bar
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=block_size):
            progress_bar.update(len(chunk))  
            file.write(chunk) 

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Downloaded file size does not match expected size")
    else:
        print(f"Download complete: {output_path}")
  
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        if not os.path.exists(output_path):
            print("downloading...")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        print(f"File downloaded successfully and saved as {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def init_model_from(url, filename):
    # if file not exists, download
    ckpt_path = Path(out_dir) / filename
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(ckpt_path):
        gr.Info('Downloading model...')
        download_file(url, ckpt_path)
        gr.Info('✅Model downloaded successfully.', duration=2)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

def respond(input, samples, model, encode, decode, max_new_tokens,temperature, top_k): # generation function
    x = (torch.tensor(encode(input), dtype=torch.long, device=device)[None, ...]) 
    with torch.no_grad():
        for k in range(samples):
            generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            output = decode(generated[0].tolist())   

            match_botoutput = re.search(r'<human>(.*?)<', output)
            match_emotion = re.search(r'<emotion>\s*(.*?)\s*<', output)
            match_context = re.search(r'<context>\s*(.*?)\s*<', output)
            response = ''
            emotion = ''
            context = ''
            if match_botoutput:
                try :
                    response = match_botoutput.group(1).replace('<endOfText>','')
                except:
                    response = match_botoutput.group(1)
            #return response, emotion, context
            return [input, response]