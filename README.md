# Autonomous LLM Prompt Orchestrator (RL Prompt Optimizer)

An AI-driven LLM prompt optimizer that uses **Reinforcement Learning (RL)** to rewrite messy, raw user queries into structured, highly optimized prompts. The goal is to maximize performance on specific downstream tasks (such as returning valid JSON structures) while minimizing prompt token length (cost efficiency).

This project is built using Python, PyTorch, Hugging Face Transformers, and Weights & Biases (WandB).

---

## Key Features

- **Local Orchestrator LLM**: Uses lightweight models (like `distilgpt2` or `gpt2`) that train easily on local consumer GPUs (e.g., 6GB VRAM) or CPUs.
- **Supervised Fine-Tuning (SFT) Baseline**: Bootstraps prompt generation from query/prompt training pairs to establish a solid initial policy.
- **Policy Gradient RL (REINFORCE)**: Optimizes the orchestrator's weights to generate rewards-driven prompts using a custom REINFORCE training loop.
- **Modulated Reward System**: Evaluates Target LLM output correctness (e.g., JSON schema adherence) and penalizes long prompts using a customizable token penalty scale.
- **Offline Simulator (Mock LLM)**: Includes a rule-based mock Target LLM, allowing developers to run the entire training and evaluation loop instantly without any API keys or network requests.
- **MLOps Tracking**: Automatic Weights & Biases dashboard initialization with a robust fallback to local CSV file logging when offline.

---

## Repository Structure

```
Prompt_Optimizer/
│
├── config.py                 # Global constants, hyperparameters, and task configurations
├── requirements.txt          # Python packages list
├── train_sft.py              # Supervised Fine-Tuning pipeline
├── train_rl.py               # Reinforcement Learning (REINFORCE) training script
├── evaluate.py               # Evaluates and prints comparative results
├── plan.md                   # Detailed project scope, RL equations, and milestone schedules
├── CHANGELOG.md              # Version tracking and change summaries
│
├── data/
│   ├── make_dataset.py       # Dataset generator (creates SFT and RL datasets)
│   ├── sft_dataset.json      # Generated SFT dataset
│   └── rl_dataset.json       # Generated RL dataset
│
├── src/
│   ├── __init__.py
│   ├── agent.py              # Orchestrator agent model wrapper
│   ├── target_llm.py         # Clients for Groq, OpenAI, or Mock Target LLM
│   ├── reward.py             # Reward scoring (task reward + token usage penalty)
│   ├── environment.py        # RL state execution loop
│   └── utils.py              # WandB & CSV fallback logger
```

---

## Quickstart Guide

### Step 1: Install Dependencies
Run the command below to install all required libraries.
```bash
python -m pip install -r requirements.txt
```

### Step 2: Generate Datasets
Create the synthetic SFT and RL JSON datasets:
```bash
python data/make_dataset.py
```

### Step 3: Run Supervised Fine-Tuning (SFT)
Bootstrap the Orchestrator LLM with target prompt behaviors to establish your Milestone 1 baseline:
```bash
python train_sft.py --epochs 3 --batch_size 4
```
This saves the fine-tuned model under `./models/sft_baseline`.

### Step 4: Run Reinforcement Learning (RL)
Train the Orchestrator using REINFORCE policy gradients to optimize prompt effectiveness and token savings:
```bash
python train_rl.py --epochs 5 --batch_size 4
```
This saves the optimized RL policy under `./models/rl_optimized`.

### Step 5: Evaluate Models
Run the evaluation suite to compare output quality and token usage between:
1. **Raw (No Optimizer)**: Passing the raw user query directly to the Target LLM.
2. **SFT Baseline**: Querying with prompts optimized by the SFT model.
3. **RL Optimized**: Querying with prompts optimized by the RL model.

```bash
python evaluate.py
```
*Note: The evaluation script runs the models in greedy decoding mode (`do_sample=False`) to produce stable, deterministic, and consistent comparative metrics.*

---

## Reward and Training Details

The system calculates total reward $R$ as:
$$R = \max(0.0, R_{\text{task}} - \text{TOKEN\_PENALTY\_SCALE} \times \text{prompt\_token\_length})$$

Where $R_{\text{task}}$ evaluates the Target LLM response text:
- **JSON Formatting Task** (default):
  - **1.0**: Perfect JSON format with required keys (`"user_id"`, `"email"`).
  - **0.9**: Valid JSON list/array (e.g. for list query).
  - **0.5**: Valid JSON format but missing required keys.
  - **0.0**: Invalid JSON format.
- **Token Reduction**: Penalizes output verbosity and conversational filler.

---

## API Credentials (Optional)

By default, `config.py` runs in `USE_MOCK_TARGET = True` for local validation. To run with actual LLMs:
1. Change `USE_MOCK_TARGET = False` in `config.py`.
2. Configure `TARGET_PROVIDER` ("groq" or "openai").
3. Set your environment variables:
```bash
set TARGET_API_KEY=gsk_your_groq_api_key_here
set WANDB_API_KEY=your_wandb_api_key_here
```

---

## Workload Allocation

- **Nhan (Infrastructure & Data)**: Manages API communication wrappers (`src/target_llm.py`), data loading, and preprocessing (`data/make_dataset.py`).
- **An (Modeling)**: Configures the base LLM Orchestrator, manages SFT baseline pipeline, and implements REINFORCE policy updates (`src/agent.py`, `train_sft.py`, `train_rl.py`).
- **Khang (MLOps & Rewards)**: Writes reward engines (`src/reward.py`), logs training progress via WandB/CSV logger (`src/utils.py`), and builds comparison benchmarks (`evaluate.py`).
