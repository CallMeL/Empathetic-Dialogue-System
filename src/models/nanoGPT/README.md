## NanoGPT
tryout [nanoGPT](https://github.com/karpathy/nanoGPT) on shakespeare dataset
## train the model
always first active the environment with `source venv/bin/activate` and install the requirements.txt. 
if there are some dependency import errors, try `pip install transformers datasets tiktoken tqdm wandb numpy`
1. download pytorch nightly for the first time
```
pip install \
  --pre torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

1. prepare the training data 
```
cd ./src/models/nanoGPT/data/shakespeare
python prepare.py
```

1. train the model
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

## test the model
###  generate text
```
cd ./src/models/nanoGPT
 python sample.py
```
###

## reference
1. [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [training nanoGPT on MacBook](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2)
