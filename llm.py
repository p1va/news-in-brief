import os
from typing import Optional

from openai import OpenAI


class OpenRouterLLM:
    def __init__(self, model: str, system_prompt: str, api_key: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def __call__(self, user_prompt: str) -> str:
        print(f"Sending request to OpenRouter (Model: {self.model})...")
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from LLM.")
            return content

        except Exception as e:
            print(f"Error communicating with OpenRouter: {e}")
            raise
