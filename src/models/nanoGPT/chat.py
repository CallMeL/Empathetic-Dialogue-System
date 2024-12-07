"""
Chat with a trained model, credits: https://github.com/VatsaDev/nanoChatGPT/blob/main/chat.py
"""
import os
import traceback
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from huggingface_hub import hf_hub_download
import shutil
import re
import sys
# -----------------------------------------------------------------------------
# if 'huggingface' then it will download the model from huggingface, 
# if 'resume' then it will resume from the out_dir

out_dir = '../trained-saved' # where trained model lives
num_samples = 1 # no samples. 1 for 1 chat at a time
max_new_tokens = 100
temperature = 0.5
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
#context="<human>Hello, how are you?<endOfText><bot>Thanks, Im good, what about you?<endOfText><human>Im great thanks, My names James, and I'm from the UK, wbu?<endOfText><bot>Hi James, I'm Conner, and im from america. <endOftext>" # a little context for better chat responses
context = ""
# -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('./configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def download_ckpt(repo_id,filename):
  local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
  custom_file_path = os.path.join(out_dir, "huggingface/ckpt-huggingface.pt")
  shutil.copy(local_file_path, custom_file_path)
  print(f"File downloaded and saved as: {custom_file_path}")

def init_model_from(path):
    ckpt_path = path
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

# init from huggingface model
def init_model(init_from):
    if init_from == 'huggingface':
        if os.path.isfile('ckpt-huggingface.pt'):
            model = init_model_from('ckpt-huggingface.pt')
        else:
            download_ckpt('VatsaDev/ChatGpt-nano', 'ckpt.pt')
            ckpt_path = './out/ckpt-huggingface.pt'
            model = init_model_from('ckpt-huggingface.pt')
    # init from our trained model   
    else:
        ckpt_path = os.path.join(out_dir,init_from,'ckpt.pt')
        print(f"Loading model from: {ckpt_path}")
        model = init_model_from(ckpt_path)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model

# -----------------------------------------------------------------------------
# gpt-2 encodings
print("loading GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def respond(input, samples, model, enable_print = True): # generation function
    x = (torch.tensor(encode(input), dtype=torch.long, device=device)[None, ...]) 
    with torch.no_grad():
        with ctx:
            for k in range(samples):
                generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

                output = decode(generated[0].tolist())   

                # match = re.search(r'<human>(.*?)<endOfText>', output)
                # match = re.search(r'<human>(.*?)\n<bot>', output)
                # wanted = match.group(1).replace('<endOfText>','')
                if enable_print:
                    # print('Robot: '+wanted)
                
                    print('----Debug: Full output--- ')
                    print(output)

                # replace context
                # output = output.replace(input,'')
                # remove any human response
                #output =  output.partition('<human>')
                # if the bot has anything left afterwards, the endOfText token is put to use
                #output_text =  output[0].rpartition('<endOftext>')
                #output_text = output[0] + output[1]
                # label removing
                #output_text = output_text.replace('<human>',' ')
                #output_text = output_text.replace('<bot>',' ')
                ## output_text = output_text.replace('<endOfText>',' ')
                # return wanted
                # return output_text

def return_single_sentence(input_sentences, init_from):
    model = init_model(init_from)
    outputs = []
    for sentence in input_sentences:
        print('Input: '+ sentence)
        start = '<bot> ' + sentence + '<human>'
        output = respond(start, num_samples, model=model, enable_print=False)
        print('Bot: '+ output)
        outputs.append(output)
    return outputs

#MARK: chat loop

if __name__ == '__main__':
    init_from = sys.argv[1] 
    model = init_model(init_from)
    while True:
        # get input from user
        start_input = input('User: ')
        start = '<bot> '+start_input+'<human>'

        # context
        # context=context+start
        
        out = respond(start, num_samples, model)
        # print(out)
        #ontext=context+out+'<endOfText>'
        #print('Bot: '+ out)

#TODO
# except KeyboardInterrupt:
#     print("\nChatting interrupted by user (Ctrl+C). Ending conversation...")
#     sys.exit(0)
# except Exception as e:
#     print("\nAn unexpected error occurred. Ending conversation...")
#     traceback.print_exc() 
#     raise