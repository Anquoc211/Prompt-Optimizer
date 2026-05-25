# data/make_dataset.py
# Functionality: Generates synthetic training datasets for both SFT (Supervised Fine-Tuning) baseline
# and RL (Reinforcement Learning) training loops, then saves them as JSON files.

import json
import os

def generate_datasets():
    # --- SFT Dataset ---
    # Contains pairs of (raw_query, optimized_prompt) for bootstrap training.
    sft_data = [
        {
            "raw_query": "make a user object with id 1 and email test@test.com",
            "optimized_prompt": "Generate a JSON object with keys: 'user_id' and 'email'. Values must be 1 and 'test@test.com' respectively. Output only JSON."
        },
        {
            "raw_query": "create a profile for john age 30 email john@gmail.com",
            "optimized_prompt": "Generate a JSON object with keys: 'user_id' (set to random) and 'email'. Values must represent john age 30 and email 'john@gmail.com'. Strictly valid JSON only."
        },
        {
            "raw_query": "give me user profile info with age 25 name alice and email alice@test.com",
            "optimized_prompt": "Generate a JSON object containing keys 'user_id' (generate integer), 'email', and 'name'. Set values to 'alice@test.com' and 'alice' with age 25. Output JSON only."
        },
        {
            "raw_query": "generate a JSON for user record contact: bob@domain.com, id 100",
            "optimized_prompt": "Create a JSON record with 'user_id' set to 100 and 'email' set to 'bob@domain.com'. Ensure output is valid JSON and no other text."
        },
        {
            "raw_query": "make user details for admin id 10 email admin@system.org",
            "optimized_prompt": "Output a valid JSON containing 'user_id' 10 and 'email' 'admin@system.org'. Output only JSON."
        },
        {
            "raw_query": "list 3 fruits",
            "optimized_prompt": "Output a JSON array containing the names of three popular fruits. Do not explain, return raw JSON."
        },
        {
            "raw_query": "convert name: david, contact: david@abc.com to json object",
            "optimized_prompt": "Generate a JSON object for david. Keys must include 'user_id' (generate one) and 'email' ('david@abc.com'). Only return JSON."
        },
        {
            "raw_query": "new user register email: reg@gmail.com, id is 50",
            "optimized_prompt": "Generate a JSON structure representing a user registration with keys 'user_id' (value 50) and 'email' ('reg@gmail.com'). Strictly valid JSON."
        }
    ]

    # --- RL Dataset ---
    # Contains a set of raw queries representing environment starting states.
    # The agent will try to mutate/optimize these queries during RL training.
    rl_data = [
        {"raw_query": "generate a user profile for user 10 email user10@test.com"},
        {"raw_query": "make a json object with id 20 and contact email: client@agency.com"},
        {"raw_query": "convert this to json: user id is 99, email is info@webservice.com"},
        {"raw_query": "create registration response for id 101, email: signup@platform.com"},
        {"raw_query": "we have a user with email customer@store.com and id 5, format as json"},
        {"raw_query": "json object contact: support@help.com, user id: 111"},
        {"raw_query": "make JSON data user: test-user, email: test-user@domain.com, id: 7"},
        {"raw_query": "generate user record for agent id 40, email agent40@agency.com"}
    ]

    # Ensure output directory exists
    os.makedirs("./data", exist_ok=True)

    sft_path = "./data/sft_dataset.json"
    rl_path = "./data/rl_dataset.json"

    with open(sft_path, "w") as f:
        json.dump(sft_data, f, indent=4)
    print(f"Saved SFT dataset ({len(sft_data)} entries) to {sft_path}")

    with open(rl_path, "w") as f:
        json.dump(rl_data, f, indent=4)
    print(f"Saved RL dataset ({len(rl_data)} entries) to {rl_path}")

if __name__ == "__main__":
    generate_datasets()
