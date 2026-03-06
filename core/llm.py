import os
from typing import Optional

from openai import OpenAI
from google import genai
from google.genai import types


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


class GeminiLLM:
    def __init__(self, model: str, system_prompt: str, api_key: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

    def __call__(self, user_prompt: str) -> str:
        print(f"Sending request to Gemini (Model: {self.model})...")
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            
            if not response.text:
                raise ValueError("Received empty response from LLM.")
            return response.text

        except Exception as e:
            print(f"Error communicating with Gemini: {e}")
            raise
