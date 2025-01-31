import re
from trl import GRPOConfig, GRPOTrainer

from data_process import get_dataset
from reward_score import compute_score



def accuracy_reward_func(prompts, completions, ground_truths, **kwargs):
    """Reward function that checks if the completion is correct."""
    solutions = [completion[0]["content"] for completion in completions]
    scores = [compute_score(solution_str=solution, ground_truth=ground_truth) for solution, ground_truth in zip(solutions, ground_truths)]
    return scores

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



train_ds, test_ds = get_dataset()

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=1)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
trainer.train()