import re, torch
import random
from trl import GRPOConfig, GRPOTrainer

from data_process import get_gsm8k_dataset
from reward_score import compute_score_gsm8k

random.seed(0)

def accuracy_reward_func(prompts, completions, **kwargs):
    """Reward function that checks if the completion is correct."""
    solutions = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get("extracted_answer")
    # print(ground_truths)
    print('accuracy reward')
    print(f'prompt:{prompts}')
    print(f'answers:{ground_truths}')
    print(f'generated_response:{solutions}')

    scores = [compute_score_gsm8k(solution_str=solution, 
                            ground_truth=ground_truth) for solution, ground_truth in zip(solutions, ground_truths)]
    return scores

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"  # allow text after think
    completion_contents = [completion[0]["content"] for completion in completions]
    # print('format reward')
    # print(completion_contents)
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

exp_id:int = 10
exp_name:str = 'test_grpo_bp_sgd'
model_name:str = "Qwen/Qwen2-0.5B"
# model_name:str = "openai-community/gpt2-medium"
train_size:int = 16
test_size:int = 2
num_exploration_steps:int = 4

train_ds = get_gsm8k_dataset(split='train')
test_ds = get_gsm8k_dataset(split='test')
print(f'datasets training:{train_ds}, validation:{test_ds}')

training_args = GRPOConfig(output_dir="qwen_multi_step-GRPO", 
                           logging_steps=1,
                           per_device_train_batch_size=2,
                           gradient_accumulation_steps=1,
                           num_generations=2,               # G in GRPO paper
                           learning_rate=1e-1,
                           max_completion_length=32,        # |o_i| in GRPO paper
                           run_name=f'{exp_id}_{exp_name}',                 # wandb logging
                           use_cpu=True,
                           num_train_epochs=1,
                           optim='sgd',
                           report_to="none",
                           num_exploration_steps = num_exploration_steps,
                        #    clip_range=0.2
                           )
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()

# Check that the params have changed
for n, param in previous_trainable_params.items():
    new_param = trainer.model.get_parameter(n)
    if torch.equal(param, new_param):
        print(f"Parameter {n} has not changed.")

