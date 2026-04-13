import os
import requests

token = "pate_your_api_key_here"

urls = [
    "https://api-inference.huggingface.co/models/google/flan-t5-base",
    "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"
]

for url in urls:
    print(f"\n--- Testing {url} ---")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": "Translate to German: Hello world", "parameters": {"max_length": 50}}
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response (text): {response.text}")
    except Exception as e:
        print(f"Failed: {e}")
