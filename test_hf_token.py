import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv('HF_TOKEN')
if not token:
    raise ValueError('HF_TOKEN not found in environment variables.')

client = InferenceClient(token=token)

try:
    result = client.text_generation(prompt="Hello, world!", model="gpt2", max_new_tokens=20)
    print("Token is valid. Response from gpt2:")
    print(result)
except Exception as e:
    print("Error during inference:")
    print(e) 