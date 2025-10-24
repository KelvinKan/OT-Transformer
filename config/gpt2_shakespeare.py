import time

out_dir = 'out-shakespeare'
eval_interval = 5 # 1000
eval_iters = 200 # 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'gpt2-shakespeare' #'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'scratch'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 4 # changed from 1 (fine-tuning) to 4
gradient_accumulation_steps = 30
max_iters = 2000
lr_decay_iters = max_iters

# default lr for GPT-2
learning_rate = 6e-4
# This min_lr is the funetuning lr
min_lr = 6e-5  # Following the NanoGPT example, learning_rate / 10 usually
# decaying learning rate
decay_lr = True

# Manually change to align with GPT-2 default setup
bias = True
