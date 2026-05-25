# evaluate.py
# Functionality: Evaluates and compares prompt quality from three configurations:
# 1. No Optimizer (Raw Query), 2. SFT Baseline, and 3. RL Optimized.
# Computes success rates, token usage, and overall reward metrics, and prints a comparative report.

import os
import json
import torch
from config import MODEL_SAVE_DIR, ACTIVE_TASK
from src.agent import OrchestratorAgent
from src.target_llm import TargetLLMClient
from src.reward import calculate_total_reward, compute_task_reward

def evaluate_all():
    print("Initializing Evaluation Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Test Queries
    test_queries = [
        "give me user record, contact is tester@qa.com, user id 12345",
        "make client data object, contact admin@corp.com, id is 8",
        "format user profile contact mail: user@domain.com, id: 9",
        "make json list of fruit items",
        "create user id 555, email dev@company.com"
    ]
    
    # 2. Initialize Target LLM
    target_client = TargetLLMClient()
    
    # 3. Load Agents
    sft_path = os.path.join(MODEL_SAVE_DIR, "sft_baseline")
    rl_path = os.path.join(MODEL_SAVE_DIR, "rl_optimized")
    
    sft_agent = None
    rl_agent = None
    
    if os.path.exists(sft_path):
        print(f"Loading SFT agent from {sft_path}...")
        sft_agent = OrchestratorAgent(model_path_or_name=sft_path, device=device)
    else:
        print("SFT agent not found. SFT evaluations will be skipped.")
        
    if os.path.exists(rl_path):
        print(f"Loading RL agent from {rl_path}...")
        rl_agent = OrchestratorAgent(model_path_or_name=rl_path, device=device)
    else:
        print("RL agent not found. RL evaluations will be skipped.")

    # 4. Run Evaluations
    results = {
        "Raw (No Optimizer)": [],
        "SFT Baseline": [],
        "RL Optimized": []
    }
    
    print("\nRunning test evaluations...")
    for q in test_queries:
        # --- Raw Query Evaluation ---
        raw_res = target_client.generate(q)
        raw_task_score = compute_task_reward(raw_res, ACTIVE_TASK)
        # Raw query token length is 0 (as it wasn't mutated by orchestrator)
        raw_reward = calculate_total_reward(raw_res, 0, ACTIVE_TASK)
        results["Raw (No Optimizer)"].append({
            "query": q,
            "mutated_prompt": q,
            "target_response": raw_res,
            "task_score": raw_task_score,
            "prompt_len": 0,
            "reward": raw_reward
        })
        
        # --- SFT Baseline Evaluation ---
        if sft_agent:
            sft_prompt, sft_len = sft_agent.generate(q, temperature=0.0)  # Greedy for evaluation
            sft_res = target_client.generate(sft_prompt)
            sft_task_score = compute_task_reward(sft_res, ACTIVE_TASK)
            sft_reward = calculate_total_reward(sft_res, sft_len, ACTIVE_TASK)
            results["SFT Baseline"].append({
                "query": q,
                "mutated_prompt": sft_prompt,
                "target_response": sft_res,
                "task_score": sft_task_score,
                "prompt_len": sft_len,
                "reward": sft_reward
            })
            
        # --- RL Optimized Evaluation ---
        if rl_agent:
            rl_prompt, rl_len = rl_agent.generate(q, temperature=0.0)  # Greedy for evaluation
            rl_res = target_client.generate(rl_prompt)
            rl_task_score = compute_task_reward(rl_res, ACTIVE_TASK)
            rl_reward = calculate_total_reward(rl_res, rl_len, ACTIVE_TASK)
            results["RL Optimized"].append({
                "query": q,
                "mutated_prompt": rl_prompt,
                "target_response": rl_res,
                "task_score": rl_task_score,
                "prompt_len": rl_len,
                "reward": rl_reward
            })

    # 5. Compile and Print Report
    print("\n==================== COMPARATIVE REPORT ====================")
    
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        avg_task = sum(r["task_score"] for r in runs) / len(runs)
        avg_len = sum(r["prompt_len"] for r in runs) / len(runs)
        avg_reward = sum(r["reward"] for r in runs) / len(runs)
        success_rate = sum(1 for r in runs if r["task_score"] >= 0.9) / len(runs) * 100
        
        summary[name] = {
            "Success Rate (%)": f"{success_rate:.1f}%",
            "Avg Task Score": f"{avg_task:.3f}",
            "Avg Prompt Token Len": f"{avg_len:.1f}",
            "Avg Overall Reward": f"{avg_reward:.3f}"
        }

    # Print summary table
    headers = ["Configuration", "Success Rate (%)", "Avg Task Score", "Avg Prompt Token Len", "Avg Overall Reward"]
    row_fmt = "{:<22} | {:<16} | {:<14} | {:<20} | {:<18}"
    print(row_fmt.format(*headers))
    print("-" * 100)
    for name, metrics in summary.items():
        print(row_fmt.format(
            name, 
            metrics["Success Rate (%)"], 
            metrics["Avg Task Score"], 
            metrics["Avg Prompt Token Len"], 
            metrics["Avg Overall Reward"]
        ))
    print("============================================================\n")

    # Display a sample case comparison
    print("Sample Case Comparison:")
    sample_q = test_queries[0]
    print(f"Raw Input: '{sample_q}'\n")
    for name, runs in results.items():
        if not runs:
            continue
        run = runs[0]
        print(f"[{name}]")
        print(f" -> Mutated Prompt:  '{run['mutated_prompt']}'")
        print(f" -> Prompt Token Len: {run['prompt_len']}")
        print(f" -> Target Response:  '{run['target_response'].strip()}'")
        print(f" -> Total Reward:     {run['reward']:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_all()
