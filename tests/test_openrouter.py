import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://sepsis-atlas-hackathon.local",
    "X-OpenRouter-Title": "Sepsis Atlas Hackathon",
  },
  json={
    # "model": "openai/gpt-4o-mini",
    "model": "anthropic/claude-3-haiku",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  }
)

print(response.json())