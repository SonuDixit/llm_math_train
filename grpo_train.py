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

'''
for multiple iterations of grpo training
ref - https://github.com/huggingface/trl/issues/2608#issuecomment-2609844003

def __init__(self, ...):
    ...
    self.train_dataset = repeat_interleave(train_dataset, self.num_grpo_iterations)  # [prompt0, prompt1] -> [prompt0, prompt0, prompt0, prompt1, prompt1, prompt1]
    # dont know why this is required
    # this is done, so that we dont need to store old_model
    # since the same data point is being repeated, the same old_log_ps is still valid
    # but this is not the right approach

def compute_loss(self, model, inputs):
    if self.step % self.num_grpo_iterations == 0: # self.num_grpo_iterations is 𝜇 in the paper
        completions = model.generate(prompts)
        self.old_log_probs = model(cat(prompts, completions))
        # detach here - since we dont update old_policy
        # note - We need to keep another copy of model, if the dataset was repeat_interleaved
        # but this is not the right approach, correctness is dependent on multiple things

    log_probs = model(cat(prompts, completions))
    log_ratio = log_probs - self.old_log_probs
    losses = min(exp(log_ratio)*advantages, clip(exp(log_ratio), 1-epsilon, 1+epsilon)*advantages)
    losses = losses - beta*kl

'''