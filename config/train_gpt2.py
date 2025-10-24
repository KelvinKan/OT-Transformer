# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

batch_size = 6
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
# Reducing number of iter to speed up training, this will make total number of tokens to be 50B
max_iters = 60000
lr_decay_iters = 60000
always_save_checkpoint = True

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
