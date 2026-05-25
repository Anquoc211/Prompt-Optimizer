# src/reward.py
# Functionality: Computes scalar reward scores based on Target LLM response validity
# (JSON structure, key requirements) and applies token penalty for prompt length optimization.

import json
from config import REWARD_TASK_WEIGHT, REWARD_TOKEN_WEIGHT, TOKEN_PENALTY_SCALE, ACTIVE_TASK

def compute_task_reward(response_text: str, task: str = ACTIVE_TASK) -> float:
    """
    Computes a scalar reward between 0.0 and 1.0 based on how well the
    Target LLM response satisfies the task criteria.
    """
    if task == "json_format":
        # Target Response must be strictly valid JSON containing specific keys: 'user_id' and 'email'
        response_text = response_text.strip()
        
        # Simple heuristic to extract JSON block if wrapped in markdown code blocks
        if response_text.startswith("```"):
            # Strip triple backticks and potential 'json' label
            lines = response_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
            
        try:
            data = json.loads(response_text)
            
            # Case 1: Valid JSON and has correct keys
            if isinstance(data, dict) and "user_id" in data and "email" in data:
                return 1.0
            # Case 2: Valid JSON array (useful for fruit lists)
            elif isinstance(data, list) and len(data) > 0:
                return 0.9
            # Case 3: Valid JSON, but missing required keys
            else:
                return 0.5
        except json.JSONDecodeError:
            # Case 4: Not valid JSON
            return 0.0
            
    elif task == "token_reduction":
        # For token reduction, we simply want the output to not be empty and be concise
        if not response_text.strip():
            return 0.0
        # Check if output seems to contain code or answer without extra chatty words
        # (e.g. less than 150 characters is good, more characters gets slight penalty)
        char_count = len(response_text)
        if char_count < 100:
            return 1.0
        elif char_count < 250:
            return 0.7
        else:
            return 0.3
            
    return 0.5

def calculate_total_reward(response_text: str, prompt_token_len: int, task: str = ACTIVE_TASK) -> float:
    """
    Combines the task reward and the token penalty.
    Total Reward = Task Reward - Penalty
    Clamped to be at least 0.0.
    """
    task_reward = compute_task_reward(response_text, task)
    
    # Calculate token length penalty (penalty increases with prompt length)
    penalty = TOKEN_PENALTY_SCALE * prompt_token_len
    
    # Combine rewards
    total_reward = task_reward - penalty
    
    # Clamp reward to [0.0, 1.0] range
    return max(0.0, min(1.0, total_reward))
