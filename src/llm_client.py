from src.config import settings
from openai import OpenAI

def get_llm_client():
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )


def chat_completion(messages, temperature=0.2, max_tokens=500):
    client = get_llm_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def analyze_text_with_context(task, transcript, context_chunks):
    context_text = "\n\n".join(
        [
            f"Источник: {chunk.get('source', 'unknown')}\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты эксперт по анализу звонков отдела продаж. "
                "Отвечай кратко, структурировано и только на русском языке."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Задача:\n{task}\n\n"
                f"Контекст из базы знаний:\n{context_text}\n\n"
                f"Транскрипт звонка:\n{transcript}"
            ),
        },
    ]
    return chat_completion(messages=messages, temperature=0.2, max_tokens=1000)
