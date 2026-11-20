import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        base_url = base_url or os.getenv("BASE_URL") or os.getenv("base_url")  # noqa: SIM112
        self.model_name = model_name or os.getenv("MODEL_NAME")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    
    async def stream_tokens(self, messages: list[dict], cancel_event=None):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            if cancel_event and cancel_event.is_set():
                break
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

if __name__ == "__main__":
    import asyncio

    async def main():
        client = LLMClient()
        print(f"Connecting to {client.model_name}...")
        messages = [{"role": "user", "content": "Tell me a 2-sentence joke."}]
        async for token in client.stream_tokens(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(main())
