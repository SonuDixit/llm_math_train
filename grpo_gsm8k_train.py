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
    # print('accuracy reward')
    # print(f'prompt:{prompts}')
    # print(f'answers:{ground_truths}')
    # print(f'generated_response:{solutions}')

    scores = [compute_score_gsm8k(solution_str=solution, 
                            ground_truth=ground_truth,
                            user_query=prompt) for prompt, solution, ground_truth in zip(prompts, solutions, ground_truths)]
    return scores

def format_reward_func(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format.
    Should start with <think> and end with <answer> tags.
    Scoring criteria:
    - 1 if <think></think> is present
    - 1 if <answer></answer> is present
    - 1 if <solution></solution> is present
    - 1 if all the above are present
    The final score is the mean of the above 4 values.
    """
    think_pattern = r"<think>.*?</think>"
    ans_pattern = r"<answer>.*?</answer>"
    sol_pattern = r"<solution>.*?</solution>"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    scores = []
    for content in completion_contents:
        has_think = bool(re.search(think_pattern, content))
        has_answer = bool(re.search(ans_pattern, content))
        has_solution = bool(re.search(sol_pattern, content))
        all_present = has_think and has_answer and has_solution
        
        # Compute the mean score
        score = sum([has_think, has_answer, has_solution, all_present]) / 4.0
        scores.append(score)
    
    return scores

exp_id:int = 14
exp_name:str = 'gsm_grpo_multi_step'
model_name:str = "Qwen/Qwen2-0.5B-Instruct"
# model_name:str = "openai-community/gpt2-medium"
# train_size:int = 16
# test_size:int = 2
num_exploration_steps:int = 4

train_ds = get_gsm8k_dataset(split='train')
test_ds = get_gsm8k_dataset(split='test')
print(f'datasets training:{train_ds}, validation:{test_ds}')

training_args = GRPOConfig(output_dir="gsm_multi_step-GRPO", 
                           logging_steps=1,
                           per_device_train_batch_size=8,
                           gradient_accumulation_steps=64,
                           num_generations=8,               # G in GRPO paper
                           learning_rate=1e-5,
                           max_completion_length=1024,        # |o_i| in GRPO paper
                           run_name=f'{exp_id}_{exp_name}',                 # wandb logging
                           use_cpu=False,
                           num_train_epochs=1,
                        #    optim='sgd',
                        #    report_to="none",
                           num_exploration_steps = num_exploration_steps,
                           clip_range=0.2
                           )
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
# previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()

# Check that the params have changed
# for n, param in previous_trainable_params.items():
#     new_param = trainer.model.get_parameter(n)
#     if torch.equal(param, new_param):
#         print(f"Parameter {n} has not changed.")

