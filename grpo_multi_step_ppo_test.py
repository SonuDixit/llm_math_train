import re, torch
import random
from trl import GRPOConfig, GRPOTrainer

from data_process import get_dataset
from reward_score import compute_score

random.seed(0)
def dummy_accuracy_reward_func(prompts, completions, **kwargs):
    """Reward function that checks if the completion is correct."""
    solutions = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get('ground_truth')
    # print(ground_truths)
    # print('accuracy reward')
    # print(solutions, ground_truths)
    scores = [compute_score(solution_str=solution, 
                            ground_truth=ground_truth) for solution, ground_truth in zip(solutions, ground_truths)]
    dummy_scores = [float(random.randint(0,1)) for _ in scores]
    print('dummy_scores:',dummy_scores)
    return dummy_scores


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

exp_id:int = 8
exp_name:str = 'test_grpo_bp_sgd'
model_name:str = "Qwen/Qwen2-0.5B"
# model_name:str = "openai-community/gpt2-medium"
train_size:int = 16
test_size:int = 2

train_ds, test_ds = get_dataset(train_size=train_size, 
                                test_size=test_size)
print(f'datasets training:{train_ds}, validation:{test_ds}')
training_args = GRPOConfig(output_dir="qwen_multi_step-GRPO", 
                           logging_steps=1,
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=1,
                           num_generations=6,               # G in GRPO paper
                           learning_rate=1e-2,
                           max_completion_length=32,        # |o_i| in GRPO paper
                           run_name=f'{exp_id}_{exp_name}',                 # wandb logging
                           use_cpu=True,
                           num_train_epochs=1,
                           optim='sgd',
                           report_to="none",
                           num_exploration_steps = 4,
                        #    clip_range=0.2
                           )
trainer = GRPOTrainer(
    model=model_name,
    # reward_funcs=[format_reward_func, accuracy_reward_func],
    reward_funcs=[format_reward_func, dummy_accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    # eval_dataset=test_ds,
)
previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()

# Check that the params have changed
for n, param in previous_trainable_params.items():
    new_param = trainer.model.get_parameter(n)
    if torch.equal(param, new_param):
        print(f"Parameter {n} has not changed.")

