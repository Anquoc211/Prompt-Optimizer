# src/target_llm.py
# Functionality: Provides clients to interact with the Target LLM (Groq, OpenAI, or a local Mock simulator).
# The MockTargetLLM inspects the prompt quality and generates responses matching the prompt effectiveness.

import json
import re
import requests
from config import TARGET_PROVIDER, TARGET_MODEL_NAME, TARGET_API_KEY, USE_MOCK_TARGET

class TargetLLMClient:
    def __init__(self):
        self.use_mock = USE_MOCK_TARGET or (TARGET_PROVIDER == "mock")
        if not self.use_mock and not TARGET_API_KEY:
            print("WARNING: Target API key not found. Falling back to Mock Target LLM.")
            self.use_mock = True

    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to the configured Target LLM and returns the text response.
        """
        if self.use_mock:
            return self._mock_generate(prompt)
        
        if TARGET_PROVIDER == "groq":
            return self._call_groq(prompt)
        elif TARGET_PROVIDER == "openai":
            return self._call_openai(prompt)
        else:
            return self._mock_generate(prompt)

    def _call_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {TARGET_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": TARGET_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0  # Keep deterministic for reward consistency
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling Groq API: {e}. Falling back to mock output.")
            return self._mock_generate(prompt)

    def _call_openai(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {TARGET_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": TARGET_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}. Falling back to mock output.")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """
        A rule-based mock generator. If the prompt contains clear constraints
        (like 'JSON', 'keys', 'user_id', 'email', 'only JSON', etc.), it generates
        well-formatted answers. Otherwise, it generates conversational text or bad JSON.
        """
        prompt_lower = prompt.lower()
        
        # Check if the prompt has high quality features (e.g. contains instructions on format, keys, and constraints)
        has_json = "json" in prompt_lower
        has_keys = "key" in prompt_lower or "field" in prompt_lower
        has_only = "only" in prompt_lower or "no other text" in prompt_lower or "strictly" in prompt_lower
        has_fields = ("user_id" in prompt_lower or "id" in prompt_lower) and "email" in prompt_lower

        # Extract whatever email or ID is mentioned in the prompt (regex)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', prompt)
        id_match = re.search(r'\b(?:id|user_id|user)\b\s*(?:is|set to|value)?\s*(\d+)', prompt, re.IGNORECASE)
        
        email = email_match.group(0) if email_match else "mock_user@example.com"
        user_id = int(id_match.group(1)) if id_match else 42

        # 1. Best Case: Prompt is highly optimized (asks for JSON, specifies keys, restricts explanation)
        if has_json and has_keys and has_only and has_fields:
            return json.dumps({"user_id": user_id, "email": email})
        
        # 2. Medium Case: Prompt asks for JSON and keys, but doesn't restrict extra text, or lacks user_id/email specific fields
        elif has_json and has_keys:
            if "fruit" in prompt_lower:
                return "Here is your JSON response:\n" + json.dumps(["apple", "banana", "cherry"]) + "\nI hope this helps!"
            return "Sure, here is the JSON data:\n" + json.dumps({"id": user_id, "email_address": email}) + "\nLet me know if you need anything else."
            
        # 3. Bad Case: Prompt is vague, conversational, or raw
        else:
            if "fruit" in prompt_lower:
                return "Sure! Here is a list of three fruits: apple, banana, and cherry. They are very delicious and healthy."
            return f"I can help you with that. The user details you provided are: ID {user_id} and email address {email}."
