"""
Chat with a trained model, credits: https://github.com/VatsaDev/nanoChatGPT/blob/main/chat.py
"""
import os
import traceback
from contextlib import nullcontext
import torch
import tiktoken
from .model import GPTConfig, GPT
from huggingface_hub import hf_hub_download
import shutil
import re
import sys

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
                if enable_print:
                    # if match_context:
                    #     context = match_context.group(1)
                    #     print('Context: '+ context)
                    if match_emotion:
                        emotion = match_emotion.group(1)
                        print('Emotion: '+ emotion)
                    if match_botoutput:
                        print('Robot: '+response)
                    print('----Debug: Full output--- ')
                    print(output)

                return response, emotion, context

def return_single_sentence(input_sentences, init_from, print_debug = True):
    model = init_model(init_from)
    outputs = []
    for sentence in input_sentences:
        if print_debug:
            print('Input: '+ sentence)
        start = '<bot> ' + sentence + '<human>'
        output = respond(start, num_samples, model=model, enable_print=False)
        if print_debug:
            print('Bot: '+ output)
        outputs.append(output)
    return outputs

#MARK: chat loop

if __name__ == '__main__':
    init_from = sys.argv[1] 
    model = init_model(init_from)
    #init_from = block_size=64/withcontext, then we need to get a pair of conversation
    if init_from != 'block_size=64/withcontext':
        while True:
            # get input from user
            start_input = input('User: ')
            start = '<bot> '+start_input+'<human>'

            # context
            # context=context+start
            
            out,_,_ = respond(start, num_samples, model)
            # print(out)
            #ontext=context+out+'<endOfText>'
            # print('Bot: '+ out)
    else:
        print("Please input a context to start the conversation")
        context = input('Context: ')
        context =  '<context>'+ context
        while True:
            user_input = input('User: ')
            start = context+ '<bot> '+user_input + '<human>'
            out,_,_ = respond(start, num_samples, model)

#TODO
# except KeyboardInterrupt:
#     print("\nChatting interrupted by user (Ctrl+C). Ending conversation...")
#     sys.exit(0)
# except Exception as e:
#     print("\nAn unexpected error occurred. Ending conversation...")
#     traceback.print_exc() 
#     raise