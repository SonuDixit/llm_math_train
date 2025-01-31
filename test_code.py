import re
import os, time
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs
import argparse

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


from data_process import get_dataset

train_ds, test_ds = get_dataset()
print(train_ds[0])


device = "cpu"
model_id = "Qwen/Qwen2.5-0.5B"

# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 15 equation?"},
# ]

messages = train_ds[0]['prompt'],

print(messages)
print('-------')

generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

pipe = pipeline("text-generation", 
                model=model_id,
                device=device,
                torch_dtype=torch.bfloat16,
                )

st_time = time.time()
gen_text = pipe(messages, **generation_args)
text = gen_text[0][0]['generated_text']
print(text)
print(f'device:{device}, Time taken:', time.time()-st_time)

from reward_score import compute_score
reward = compute_score(solution_str=text, 
                       ground_truth=train_ds[0]['reward_model']['ground_truth'], 
                       method='strict', 
                       format_score=0.1, 
                       score=1.)
print(reward)