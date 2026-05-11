import os

from openai import OpenAI


base_url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/v1"
api_key = os.environ.get("OPENAI_API_KEY", "local-test-key")
model = os.environ.get("OPENAI_MODEL", "qwen3.6-27b")

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are concise and practical."},
        {"role": "user", "content": "Reply with one sentence confirming the API works."},
    ],
    max_tokens=80,
    temperature=0.2,
)

message = response.choices[0].message
print(message.content or getattr(message, "reasoning_content", ""))
