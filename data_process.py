# ref - https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs
import argparse


def get_dataset(train_size: int = 327680, test_size: int = 1024) -> Tuple[Dataset, Dataset]:
    # checking the online version
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    print(raw_dataset[0])
    print(raw_dataset[1])

    print(raw_dataset)

    def make_prefix(dp, template_type):
        target = dp['target']
        numbers = dp['nums']
        # NOTE: also need to change reward_score/countdown.py
        if template_type == 'base':
            """This works for any base model"""
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
    Assistant: Let me solve this step by step.
    <think>"""
        elif template_type == 'qwen-instruct':
            """This works for Qwen Instruct Models"""
            prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
        return prefix


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_dir', default='~/data/countdown')
    # parser.add_argument('--hdfs_dir', default=None)
    # parser.add_argument('--num_samples', type=int, default=100000)
    # parser.add_argument('--num_operands', type=int, default=6)
    # parser.add_argument('--max_target', type=int, default=1000)
    # parser.add_argument('--min_number', type=int, default=1)
    # parser.add_argument('--max_number', type=int, default=100)
    # parser.add_argument('--train_size', type=int, default=327680)
    # parser.add_argument('--test_size', type=int, default=1024)
    # parser.add_argument('--template_type', type=str, default='base')

    # args = parser.parse_args()

    data_source = 'countdown'
    TRAIN_SIZE = train_size
    TEST_SIZE = test_size

    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type='base')
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }
            # data = {
            #     "data_source": data_source,
            #     "prompt": [{
            #         "role": "user",
            #         "content": question,
            #     }],
            #     "ability": "math",
            #     "reward_model": {
            #         "style": "rule",
            #         "ground_truth": solution
            #     },
            #     "extra_info": {
            #         'split': split,
            #         'index': idx,
            #     }
            # }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "ground_truth": solution,
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    return train_dataset, test_dataset

def get_gsm8k_dataset(split: str = 'train') -> Dataset:
    sys_prompt = """You are an helpful Assistant with excellent reasoning ability. When the user asks the question and the assistant solves the problem by reasoning in a step by step process and then provides the user with the answer. Always respond in the following format:
                    <think> {your step by step reasoning process here} </think>
                    <answer> {answer here} </answer>
                    """
    def format_user_question(user_question: str) -> str:
        # prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        # User: {user_question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. 
        # Assistant: Let me solve this step by step.<think>"""
        
        prefix = """You are an helpful Assistant with excellent reasoning ability. When the user asks the question and the assistant solves the problem by reasoning in a step by step process and then provides the user with the answer. Always respond in the following format:
                    <think> {your step by step reasoning process here} </think>
                    <answer> {answer here} </answer>
                    """
        prefix = prefix + user_question
        return prefix
    def extract_hash_answer(text: str | None) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")
    
    
    ds = load_dataset("openai/gsm8k", "main", split=split)
    print(ds)
    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["answer"],
        "extracted_answer": extract_hash_answer(x["answer"]),
        "prompt": [{"role":"system",
                    "content":sys_prompt,
                    },
                    {"role": "user",
                    "content": x["question"],
                    }]
         }
        )
    return ds





if __name__ == "__main__":
    
    pass