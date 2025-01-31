import re
from trl import GRPOConfig, GRPOTrainer

from data_process import get_dataset
from reward_score import compute_score


def accuracy_reward_func(prompts, completions, **kwargs):
    """Reward function that checks if the completion is correct."""
    solutions = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get('ground_truth')
    # print(ground_truths)
    scores = [compute_score(solution_str=solution, 
                            ground_truth=ground_truth) for solution, ground_truth in zip(solutions, ground_truths)]
    return scores

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

exp_id:int = 2
exp_name:str = 'exp'

train_ds, test_ds = get_dataset()
print(f'datasets training:{train_ds}, validation:{test_ds}')
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", 
                           logging_steps=4,
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=32,
                           num_generations=16,               # G in GRPO
                           learning_rate=1e-5,
                           run_name=f'{exp_id}_{exp_name}'                 # wandb logging
                           )
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
trainer.train()