# train_sft.py
# Functionality: Loads raw query/optimized prompt pairs and performs supervised fine-tuning (SFT)
# on the causal LLM Orchestrator to bootstrap its prompt generation capabilities.

import argparse
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from config import SFT_EPOCHS, SFT_BATCH_SIZE, SFT_LR, MODEL_SAVE_DIR
from src.agent import OrchestratorAgent
from src.utils import ExperimentLogger

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path {data_path} does not exist. Run data/make_dataset.py first.")
            
        with open(data_path, "r") as f:
            self.raw_data = json.load(f)
            
        self.items = []
        for entry in self.raw_data:
            # Format: "Optimize this query: {raw_query}\nOptimized prompt: {optimized_prompt}<|endoftext|>"
            full_text = f"Optimize this query: {entry['raw_query']}\nOptimized prompt: {entry['optimized_prompt']}{tokenizer.eos_token}"
            
            encodings = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            # Causal LM training: labels are same as input_ids.
            # Hugging Face causal LMs automatically shift labels internally.
            labels = list(input_ids)
            
            # Mask the instruction prefix so loss is only calculated on the generated prompt:
            # We locate where "Optimized prompt: " ends.
            prompt_prefix = f"Optimize this query: {entry['raw_query']}\nOptimized prompt: "
            prefix_ids = tokenizer(prompt_prefix, truncation=True, max_length=max_length)["input_ids"]
            prefix_len = len(prefix_ids)
            
            # Mask label tokens before prompt prefix with -100 (which PyTorch CrossEntropy ignores)
            for i in range(min(prefix_len, len(labels))):
                labels[i] = -100
                
            self.items.append({
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def run_sft(epochs=SFT_EPOCHS, batch_size=SFT_BATCH_SIZE, lr=SFT_LR):
    print("Starting SFT Baseline Training...")
    
    # Initialize Agent
    agent = OrchestratorAgent()
    
    # Load dataset
    dataset_path = "./data/sft_dataset.json"
    try:
        train_dataset = SFTDataset(dataset_path, agent.tokenizer)
    except FileNotFoundError:
        print("Dataset not found. Generating synthetic dataset now...")
        from data.make_dataset import generate_datasets
        generate_datasets()
        train_dataset = SFTDataset(dataset_path, agent.tokenizer)
        
    # Setup Logger
    logger = ExperimentLogger(
        run_name=f"sft-baseline-run-{epochs}ep",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "orchestrator": agent.model.name_or_path
        }
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./sft_results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb" if logger.use_wandb else "none",
        fp16=torch.cuda.is_available()  # Use mixed precision if GPU is available
    )
    
    # Define Trainer
    trainer = Trainer(
        model=agent.model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # Train
    train_result = trainer.train()
    
    # Log final training loss
    loss = train_result.training_loss
    logger.log_metrics(epoch=epochs, step=1, loss=loss, mean_reward=0.5, mean_prompt_len=0.0)
    
    # Save the fine-tuned model
    save_path = os.path.join(MODEL_SAVE_DIR, "sft_baseline")
    agent.save(save_path)
    logger.finish()
    print("SFT Baseline Training Completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=SFT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=SFT_LR)
    args = parser.parse_args()
    
    run_sft(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
