# NanoGPT
[nanoGPT](https://github.com/karpathy/nanoGPT)
## about NanoGPT and transformer
A transformer based language model
GPT: Generatively Pretrained Transformer
* How encoder and decoder works in simple words
  * encoder: take a string -> output a list of integers
    * in the code the [tiktoken](https://github.com/openai/tiktoken) from openAI is used
  * decoder: take a list of integers -> out put a string

* Self-Attention
  * Q: what am I looking for
  * K: what am I containing
  * W: dot product of Q*K (the attention score)
  ```
  # class CausalSelfAttention in model.py
  # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
       
  ```
## the loss
`loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)` in `model.py`

## the training part
`optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)`

## Run the code
### train the model
always first active the environment with `source venv/bin/activate` and install the requirements.txt. 
if there are some dependency import errors, try `pip install transformers datasets tiktoken tqdm wandb numpy`
1. download pytorch nightly for the first time
```
pip install \
  --pre torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

2. prepare the training data 
```
cd ./src/models/nanoGPT/data/shakespeare
python prepare.py
```

3. train the model
`--device` can be set to `cpu` or specially `mps` on MacBooks. 
`max_iters` in the train.py is set to `600000` , it runs forever~ use `^ C` to stop the training at anytime.
```
cd ./src/models/nanoGPT

time python train.py \
  --dataset=shakespeare \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=mps
```
When the training starts, hit `^ A` so later we can copy all the logs to this [website](https://observablehq.com/@simonw/plot-loss-from-nanogpt), then get the log graph. (we will definitely improve the logging later )

### test the model
###  generate text
```
cd ./src/models/nanoGPT
 python sample.py
```
###

## reference
1. [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [training nanoGPT on MacBook](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2)
3. [Let's build GPT: from scratch, in code, spelled out. (author of nanoGPT)](https://youtu.be/kCc8FmEb1nY?si=XA_iMh2jns5vPHN5)