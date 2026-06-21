import json
import re

from openai import OpenAI

from src.config import settings


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


def extract_json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain JSON")

    return json.loads(match.group(0))


def normalize_basic_analysis(data):
    return {
        "summary": str(data.get("summary", "")),
        "call_result": str(data.get("call_result", "unknown")),
        "total_score": int(data.get("total_score", 0)),
        "strengths": data.get("strengths") or [],
        "weaknesses": data.get("weaknesses") or [],
        "recommendations": data.get("recommendations") or [],
        "script_compliance": data.get("script_compliance") or [],
        "objections": data.get("objections") or [],
    }


def analyze_sales_call_basic(transcript, context_chunks):
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
                "Отвечай только валидным JSON без markdown, пояснений и текста вокруг JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Проанализируй звонок отдела продаж по транскрипту и базе знаний.\n"
                "Верни строго JSON такой структуры:\n"
                "{\n"
                '  "summary": "краткое содержание звонка",\n'
                '  "call_result": "follow_up_required",\n'
                '  "total_score": 72,\n'
                '  "strengths": ["сильная сторона"],\n'
                '  "weaknesses": ["слабая сторона"],\n'
                '  "recommendations": ["рекомендация"],\n'
                '  "script_compliance": [\n'
                '    {"step": "Приветствие", "status": "done", "comment": "комментарий"}\n'
                "  ],\n"
                '  "objections": [\n'
                '    {"type": "Цена", "client_phrase": "фраза клиента", "manager_response_quality": "weak", "recommendation": "рекомендация"}\n'
                "  ]\n"
                "}\n\n"
                "Требования:\n"
                "- total_score должен быть числом от 0 до 100.\n"
                "- call_result выбери коротким snake_case значением.\n"
                "- script_compliance должен покрывать этапы скрипта продаж.\n"
                "- Если возражений нет, верни пустой список objections.\n\n"
                f"Контекст из базы знаний:\n{context_text}\n\n"
                f"Транскрипт звонка:\n{transcript}"
            ),
        },
    ]
    response_text = chat_completion(messages=messages, temperature=0.1, max_tokens=1500)
    return normalize_basic_analysis(extract_json_from_text(response_text))
