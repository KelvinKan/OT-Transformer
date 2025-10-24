# Evaluate saved models

import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from torch.nn import functional as F

import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
from sacremoses import MosesTokenizer
from transformers import AutoTokenizer

from torchtext.data.metrics import bleu_score
import bert_score
from rouge_score import rouge_scorer

# Suppress warning from bert computation
import logging
# Suppress transformers library warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Suppress specific UserWarning related to size_average and reduce
import warnings
warnings.filterwarnings("ignore", message=".*size_average and reduce args will be deprecated.*")

import json
from collections import defaultdict
import tiktoken

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

lmbda = 0.1
cts = True
steps = 10
ln1_on = True
ln2_on = True
lnf_on = True
tokenization = "Moses"

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

print("config: ", config)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for later use in torch.autocast
print("device: ", device)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

enc = tiktoken.get_encoding("gpt2")

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
def compute_rouge_for_sentences(targets, predictions):
    # return the average fmeasure for rouge1, rouge2, rougeL

    average_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    count = 0
    for targ, pred in zip(targets, predictions):
        score = scorer.score(targ[0], pred)
        count += 1
        for key in average_scores:
            average_scores[key] += score[key].fmeasure

    for key in average_scores:
        average_scores[key] /= count

    return average_scores

def compute_score():
    with torch.no_grad():

        metrics_name = ['val_loss', 'Perplexity', 'Bleu', 'BertP', 'BertR', 'BertF1', 'Rouge1', 'Rouge2', 'RougeL']
        model.eval()

        metrics = {key: torch.zeros(eval_iters) for key in metrics_name}
        split = 'val'

        for k in range(eval_iters):
            print("iteration: ", k)
            X, Y = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, Y)

            # The logits have shape [batch_size, block_size, meta_vocab_size]
            # The targets have shape [batch_size, block_size]

            # Y is targets
            loss_per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduce=False,
                                             ignore_index=-1)
            loss_per_batch = loss_per_token.view(*Y.shape).mean(dim=1)

            perplexity = torch.exp(loss_per_batch).mean()

            predictions = torch.argmax(logits, dim=-1)

            # bleu can take multiple candidates targets, so need to unsqueeze and make the target 3d for computation
            # predicted_sequences, target_sequences = predictions.tolist(), Y.tolist()
            # from torchtext.data.metrics import bleu_score
            # bleu_score(predicted_sequences, target_sequences)

            predicted_sentences = enc.decode_batch([seq.tolist() for seq in predictions])
            target_texts = enc.decode_batch([seq.tolist() for seq in Y])
            # input_texts = enc.decode_batch([seq.tolist() for seq in X])
            target_sentences = [[seq] for seq in target_texts]

            # Reference: https://iq.opengenus.org/tokenization-in-nlp/
            # Penn Tree tokenization
            if tokenization == "Penn":
                tokenizer = word_tokenize
            elif tokenization == "Moses":
                mt = MosesTokenizer()
                tokenizer = mt.tokenize
            elif tokenization == "pretrained":
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                tokenizer = tokenizer.tokenize

            list_predicted_words = [tokenizer(sentence) for sentence in predicted_sentences]
            list_target_words = [[tokenizer(*sentence)] for sentence in target_sentences]

            computed_bleu = bleu_score(list_predicted_words, list_target_words)
            BertP, BertR, BertF1 = bert_score.score(predicted_sentences, target_sentences, lang='en', verbose=False, device=device)

            BertP, BertR, BertF1 = BertP.mean(), BertR.mean(), BertF1.mean()

            computed_rouge = compute_rouge_for_sentences(target_sentences, predicted_sentences)

            metrics['val_loss'][k] = loss
            metrics['Perplexity'][k] = perplexity
            metrics['Bleu'][k] = computed_bleu
            metrics['BertP'][k], metrics['BertR'][k], metrics['BertF1'][k] = BertP, BertR, BertF1
            metrics['Rouge1'][k], metrics['Rouge2'][k], metrics['RougeL'][k] = computed_rouge['rouge1'], \
            computed_rouge['rouge2'], computed_rouge['rougeL']

        metrics['val_loss'] = metrics['val_loss'].mean().item()
        metrics['Perplexity'] = metrics['Perplexity'].mean().item()
        metrics['Bleu'] = metrics['Bleu'].mean().item()
        metrics['BertP'], metrics['BertR'], metrics['BertF1'] = metrics['BertP'].mean().item(), metrics['BertR'].mean().item(), metrics[
            'BertF1'].mean().item()
        metrics['Rouge1'], metrics['Rouge2'], metrics['RougeL'] = metrics['Rouge1'].mean().item(), metrics['Rouge2'].mean().item(), \
            metrics['RougeL'].mean().item()
    # model.train()
    return metrics


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, cts=cts, steps=steps,
                  ln1_on=ln1_on, ln2_on=ln2_on, lnf_on=lnf_on, lmbda=lmbda)


results_file = os.path.join(out_dir, "evaluated_results.json")
# Load previously evaluated results if the file exists
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        evaluated_results = json.load(f)
else:
    evaluated_results = {}

# results = []

for filename in os.listdir(out_dir):
    seed_everything(1337)
    if filename.endswith('.pt'):
        ckpt_path = os.path.join(out_dir, filename)  # Get the full path
    else:
        # Pass files that are not checkpoints .pt
        continue

    # Check if the model is already evaluated
    if filename in evaluated_results:
        print(f"Skipping {filename}, already evaluated.")
        continue

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout',
              'cts', 'steps', 'ln1_on', 'ln2_on', 'lnf_on', 'lmbda', 'random_seed']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    print("loading model: {}".format(filename))
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # so that the checkpoint will have the right value

    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    metrics = compute_score()


    evaluated_results[filename] = [model_args['random_seed'], model_args['cts'], model_args['steps'], model_args['lmbda'], model_args['ln1_on'], model_args['ln2_on'],
         model_args['lnf_on'],
        metrics['val_loss'],
         metrics['Perplexity'], metrics['Bleu'], metrics['Rouge1'], metrics['Rouge2'], metrics['RougeL'],
         metrics['BertP'], metrics['BertR'], metrics['BertF1']]

    # Save the updated results to file
    with open(results_file, "w") as f:
        json.dump(evaluated_results, f, indent=4)

# Average over trials
aggregated_results = defaultdict(list)

# Group by model args excluding 'random_seed'
for filename, values in evaluated_results.items():
    key = tuple(values[1:7])  # Exclude random_seed, keep other model args
    metrics = values[7:]  # Extract metrics
    aggregated_results[key].append(metrics)

# Modify aggregated_results in place to store the final averaged results
for key in aggregated_results:
    metrics_list = aggregated_results[key]
    num_trials = len(metrics_list)
    avg_metrics = np.mean(metrics_list, axis=0).tolist()
    aggregated_results[key] = [num_trials] + list(key) + avg_metrics

header = [
    "no_trials", "CTS", "Steps", "Lambda", "LN1 On", "LN2 On", "LNF On",
    "val_loss",
    "Perplexity", "Bleu", "Rouge1", "Rouge2", "RougeL",
    "BertP", "BertR", "BertF1"
]


print("{:<10} {:<5} {:<7} {:<15} {:<7} {:<7} {:<7} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*header))


for key in sorted(aggregated_results.keys()):
    formatted_result = [
        f"{value:.6f}" if isinstance(value, float) else str(value)
        for value in aggregated_results[key]
    ]

    print("{:<10} {:<5} {:<7} {:<15} {:<7} {:<7} {:<7} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*formatted_result))
