import re, torch
import random
from trl import GRPOConfig, GRPOTrainer

from data_process import get_dataset
from reward_score import compute_score


def accuracy_reward_func(prompts, completions, **kwargs):
    """Reward function that checks if the completion is correct."""
    solutions = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get('ground_truth')
    # print(ground_truths)
    # print('accuracy reward')
    # print(solutions, ground_truths)
    scores = [compute_score(solution_str=solution, 
                            ground_truth=ground_truth) for solution, ground_truth in zip(solutions, ground_truths)]
    return scores

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    # print('format reward')
    # print(completion_contents)
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

exp_id:int = 10
exp_name:str = 'first_exp'
model_name:str = "Qwen/Qwen2-0.5B"
train_size:int = 327680
test_size:int = 1024

train_ds, test_ds = get_dataset(train_size=train_size, 
                                test_size=test_size)
print(f'datasets training:{train_ds}, validation:{test_ds}')

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", 
                           logging_steps=1,
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=32,
                           num_generations=16,               # G in GRPO paper
                           learning_rate=1e-5,
                           max_completion_length=512,        # |o_i| in GRPO paper
                           run_name=f'{exp_id}_{exp_name}',  # wandb logging
                           use_cpu=False,                    # cpu for local mps device
                           num_train_epochs=1,
                           )
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
trainer.train()
