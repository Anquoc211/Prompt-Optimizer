# src/agent.py
# Functionality: Defines the Orchestrator class wrapping a Hugging Face Causal LM.
# Handles formatting inputs, generating mutated prompts, and tokenization.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ORCHESTRATOR_MODEL_NAME, MAX_PROMPT_LEN

class OrchestratorAgent:
    def __init__(self, model_path_or_name: str = ORCHESTRATOR_MODEL_NAME, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Orchestrator Agent model '{model_path_or_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_path_or_name).to(self.device)
        
        # Ensure padding token is set (distilgpt2 doesn't have a default pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_input_text(self, raw_query: str) -> str:
        """
        Formats the input query into the orchestrator prompt.
        """
        return f"Optimize this query: {raw_query}\nOptimized prompt: "

    def generate(self, raw_query: str, max_new_tokens: int = MAX_PROMPT_LEN, temperature: float = 0.7) -> tuple[str, int]:
        """
        Generates an optimized prompt from a raw query.
        Returns a tuple: (generated_prompt_string, token_length)
        """
        input_text = self.get_input_text(raw_query)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # We generate text, ensuring we only sample from the new tokens
        self.model.eval()
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        # Extract only the generated prompt part (exclude input query prefix)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        
        # Decode
        generated_prompt = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Split on newline to avoid runaway generations
        if "\n" in generated_prompt:
            generated_prompt = generated_prompt.split("\n")[0].strip()
            
        # If the generated prompt is empty, fallback to the raw query
        if not generated_prompt:
            generated_prompt = raw_query
            
        return generated_prompt, len(generated_ids)
        
    def save(self, save_path: str):
        """
        Saves the orchestrator model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
