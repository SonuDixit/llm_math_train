from reward_score import compute_score_gsm8k
import re

user_query = "hello"
solution_str = '''
<think> I need to add 20% of 20% of 20% of ads, then multiply by 100 to get the percentage </think>
209  <solution> 20% of 20% of 20% = .20 * .20 * .20 = 0.0004 </solution>
210  <answer> 0.04 </answer>
'''
ground_truth = {'target':0.04}

reward = compute_score_gsm8k(solution_str=solution_str,
              ground_truth=ground_truth,
              user_query=user_query)
print(reward)


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        print('result eval equation here :',result)
        return result
    except Exception as e:
        return None

print(evaluate_equation('0.04'))