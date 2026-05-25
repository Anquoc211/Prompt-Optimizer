# train_rl.py
# Functionality: Trains the Orchestrator agent using Reinforcement Learning (REINFORCE policy gradient).
# Mutates prompts, queries Target LLM, computes rewards, and performs policy gradient updates.

import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
from config import RL_EPOCHS, RL_BATCH_SIZE, RL_LR, MODEL_SAVE_DIR, ACTIVE_TASK
from src.agent import OrchestratorAgent
from src.environment import PromptOptimizationEnv
from src.utils import ExperimentLogger

def run_rl(epochs=RL_EPOCHS, batch_size=RL_BATCH_SIZE, lr=RL_LR):
    print("Starting RL Policy Gradient Optimization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Agent
    # Try to load SFT baseline if available, otherwise load base model
    sft_model_path = os.path.join(MODEL_SAVE_DIR, "sft_baseline")
    if os.path.exists(sft_model_path):
        print(f"Loading SFT baseline model from {sft_model_path}")
        agent = OrchestratorAgent(model_path_or_name=sft_model_path, device=device)
    else:
        print("SFT baseline model not found. Starting RL from base pre-trained model.")
        agent = OrchestratorAgent(device=device)

    # Ensure padding token is set
    if agent.tokenizer.pad_token is None:
        agent.tokenizer.pad_token = agent.tokenizer.eos_token
        agent.model.config.pad_token_id = agent.model.config.eos_token_id

    # 2. Load RL dataset (raw queries)
    dataset_path = "./data/rl_dataset.json"
    if not os.path.exists(dataset_path):
        print("RL Dataset not found. Generating datasets now...")
        from data.make_dataset import generate_datasets
        generate_datasets()
        
    with open(dataset_path, "r") as f:
        rl_data = json.load(f)
        
    # 3. Setup Env and Logger
    env = PromptOptimizationEnv()
    logger = ExperimentLogger(
        run_name=f"rl-ppo-run-{epochs}ep",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "orchestrator": agent.model.name_or_path,
            "task": ACTIVE_TASK
        }
    )
    
    # 4. Initialize Optimizer and Reward Baseline
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=lr)
    reward_baseline = 0.5  # Initial baseline for reward variance reduction
    
    # 5. Training Loop
    global_step = 0
    agent.model.train()
    
    for epoch in range(epochs):
        # Shuffle queries each epoch
        random.shuffle(rl_data)
        
        # Batching
        for i in range(0, len(rl_data), batch_size):
            batch_queries = rl_data[i:i+batch_size]
            if len(batch_queries) < batch_size:
                continue # Skip partial batches for stable tensor sizing
                
            optimizer.zero_grad()
            
            # Rollout: Generate prompts, query Target LLM, compute rewards
            batch_results = []
            for item in batch_queries:
                raw_q = item["raw_query"]
                step_res = env.step(agent, raw_q, temperature=0.7)
                batch_results.append(step_res)
                
            # Logging prompt evaluations to WandB Table & CSV
            for res in batch_results:
                logger.log_prompts(
                    raw_query=res["raw_query"],
                    optimized_prompt=res["optimized_prompt"],
                    target_response=res["target_response"],
                    reward=res["reward"]
                )
                
            # Extract values for Policy Gradient calculation
            rewards = [res["reward"] for res in batch_results]
            mean_reward = sum(rewards) / len(rewards)
            mean_prompt_len = sum([res["prompt_token_len"] for res in batch_results]) / len(batch_results)
            
            # Update moving reward baseline
            reward_baseline = 0.9 * reward_baseline + 0.1 * mean_reward
            
            # Policy gradient computation:
            # We must compute log-probs of the generated tokens under the current model parameters.
            loss = 0.0
            
            for res, reward in zip(batch_results, rewards):
                raw_q = res["raw_query"]
                opt_prompt = res["optimized_prompt"]
                
                # Format combined text
                input_prefix = agent.get_input_text(raw_q)
                full_text = input_prefix + opt_prompt + agent.tokenizer.eos_token
                
                # Tokenize
                encodings = agent.tokenizer(full_text, return_tensors="pt").to(device)
                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]
                
                # Get lengths
                prefix_ids = agent.tokenizer(input_prefix, return_tensors="pt")["input_ids"]
                prefix_len = prefix_ids.shape[1]
                total_len = input_ids.shape[1]
                
                if total_len <= prefix_len:
                    continue  # Safety check if no new tokens were generated
                
                # Forward pass to get logits
                outputs = agent.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape: (1, seq_len, vocab_size)
                
                # Shift logits and labels for Causal LM training (next token prediction)
                shift_logits = logits[0, :-1, :]  # shape: (seq_len - 1, vocab_size)
                shift_labels = input_ids[0, 1:]   # shape: (seq_len - 1)
                
                # We only want the loss / log-probabilities of the generated prompt part.
                # The generated prompt starts at index prefix_len - 1 in shifted labels.
                gen_logits = shift_logits[prefix_len-1:]
                gen_labels = shift_labels[prefix_len-1:]
                
                # Calculate negative log likelihood (which is cross-entropy loss)
                # log P(t_i) = -CE(logits_i, label_i)
                ce_loss = F.cross_entropy(gen_logits, gen_labels, reduction="sum")
                log_prob = -ce_loss  # Log probability of generating the prompt sequence
                
                # REINFORCE loss: - (reward - baseline) * log_prob
                advantage = reward - reward_baseline
                sample_loss = -advantage * log_prob
                
                # Accumulate loss
                loss += sample_loss
                
            loss = loss / batch_size  # Mean loss
            
            # Backpropagation (only if we generated something and loss is non-zero)
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_val = loss.item()
            else:
                loss_val = 0.0
                
            # Log metrics
            logger.log_metrics(
                epoch=epoch,
                step=global_step,
                loss=loss_val,
                mean_reward=mean_reward,
                mean_prompt_len=mean_prompt_len
            )
            print(f"Epoch {epoch+1}/{epochs} | Batch {i//batch_size+1} | Loss: {loss_val:.4f} | Mean Reward: {mean_reward:.4f} | Mean Length: {mean_prompt_len:.1f}")
            global_step += 1

    # Save RL model
    save_path = os.path.join(MODEL_SAVE_DIR, "rl_optimized")
    agent.save(save_path)
    logger.finish()
    print("RL Policy Gradient Optimization Completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=RL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=RL_LR)
    args = parser.parse_args()
    
    run_rl(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
