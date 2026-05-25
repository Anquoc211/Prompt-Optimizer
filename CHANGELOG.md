# Changelog

All notable changes to the LLM Prompt Optimizer RL project will be documented in this file.

## [1.0.0] - 2026-05-25

### Added
- **`requirements.txt`**: Defined core library dependencies (`torch`, `transformers`, `accelerate`, `wandb`, `tqdm`, `requests`, `openai`).
- **`config.py`**: Global configuration file containing hyperparameters, API credentials, paths, and reward weights. Supports task switching between JSON formatting and token minimization.
- **`data/make_dataset.py`**: Synthetic dataset generator creating SFT dataset (`sft_dataset.json`) and RL training dataset (`rl_dataset.json`).
- **`src/target_llm.py`**: Client module managing API requests to Groq/OpenAI target models. Integrates `MockTargetLLM` for offline baseline development.
- **`src/reward.py`**: Reward function module. Implements JSON validation checks (presence of required keys) and a prompt token length penalty.
- **`src/environment.py`**: RL environment interface (`PromptOptimizationEnv`) connecting agent mutations, target execution, and reward returns.
- **`src/agent.py`**: Wrapper for the local causal LLM Orchestrator (e.g., `distilgpt2`), providing custom instruction formatting, greedy/sampled prompt generation, and tokenization.
- **`src/utils.py`**: Logging utility (`ExperimentLogger`) handling Weights & Biases (WandB) reporting with automatic fallback to local CSV files if offline.
- **`train_sft.py`**: Fine-tuning script to train the Orchestrator on raw/optimized query pairs using Hugging Face `Trainer`.
- **`train_rl.py`**: RL optimization loop using custom REINFORCE policy gradient with a moving average reward baseline.
- **`evaluate.py`**: Evaluation script comparing Raw Queries, SFT baseline, and RL optimized prompt generators across success rates, token usage, and reward metrics.

### Fixed
- **`train_sft.py`**: Removed `overwrite_output_dir` from `TrainingArguments` to prevent compatibility crashes in newer `transformers` versions.
- **`src/agent.py`**: Conditionally set `do_sample=False` when `temperature <= 0.0` to support greedy decoding in evaluation without throwing a `ValueError`.
