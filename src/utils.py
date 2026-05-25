# src/utils.py
# Functionality: Provides metric logging utilities. Handles Weights & Biases (WandB) integration
# and features automatic local CSV file fallback logging when WandB is unavailable or disabled.

import os
import csv
import datetime
from config import PROJECT_NAME, LOG_DIR, WANDB_API_KEY

class ExperimentLogger:
    def __init__(self, run_name: str, config: dict):
        self.run_name = run_name
        self.config = config
        self.use_wandb = False
        
        # Create local log directory
        os.makedirs(LOG_DIR, exist_ok=True)
        self.csv_path = os.path.join(LOG_DIR, f"{run_name}_metrics.csv")
        self.table_path = os.path.join(LOG_DIR, f"{run_name}_prompts.csv")
        
        # Initialize CSV log headers
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Epoch", "Step", "Loss", "Mean_Reward", "Mean_Prompt_Len"])
            
        with open(self.table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Raw_Query", "Optimized_Prompt", "Target_Response", "Reward"])

        # Try initializing WandB
        # If API key is in config or environment, login
        if WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            
        try:
            import wandb
            # Test if logged in or can initialize offline/online
            wandb.init(
                project=PROJECT_NAME,
                name=run_name,
                config=config,
                dir=LOG_DIR,
                settings=wandb.Settings(start_method="thread")
            )
            self.use_wandb = True
            self.wandb = wandb
            print("Successfully initialized WandB logging.")
        except Exception as e:
            print(f"WandB initialization failed/skipped: {e}.")
            print(f"Logging will save locally to:\n- Metrics: {self.csv_path}\n- Prompts: {self.table_path}")

    def log_metrics(self, epoch: int, step: int, loss: float, mean_reward: float, mean_prompt_len: float):
        """
        Logs numerical metrics to WandB or CSV.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Local CSV write
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, epoch, step, loss, mean_reward, mean_prompt_len])
        except Exception as e:
            print(f"Failed to log metrics to CSV: {e}")

        # WandB write
        if self.use_wandb:
            self.wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "mean_reward": mean_reward,
                "mean_prompt_len": mean_prompt_len
            })

    def log_prompts(self, raw_query: str, optimized_prompt: str, target_response: str, reward: float):
        """
        Logs individual prompt optimization runs to CSV and WandB Tables.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Local CSV write
        try:
            with open(self.table_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, raw_query, optimized_prompt, target_response, reward])
        except Exception as e:
            print(f"Failed to log prompts to CSV: {e}")

        # WandB Table log (we append to a local list or log as a single log step)
        if self.use_wandb:
            table = self.wandb.Table(columns=["Raw Query", "Optimized Prompt", "Target Response", "Reward"])
            table.add_data(raw_query, optimized_prompt, target_response, reward)
            self.wandb.log({"Prompt_Evaluations": table})

    def finish(self):
        if self.use_wandb:
            self.wandb.finish()
            print("WandB session finished successfully.")
