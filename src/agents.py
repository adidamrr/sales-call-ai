import json

try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

from src import llm_client, rag


class CallAnalysisState(TypedDict):
    call_id: int
    transcript: str
    rag_context: dict
    summary: dict
    classification: dict
    script_check: dict
    objections: dict
    scoring: dict
    recommendations: dict
    final_report: dict


def parse_llm_json(text):
    return llm_client.extract_json_from_text(text)


def format_chunks(chunks):
    return "\n\n".join(
        [
            f"Источник: {chunk.get('source', 'unknown')}\n{chunk.get('text', '')}"
            for chunk in chunks
        ]
    )


def ask_json_agent(system_prompt, user_prompt, max_tokens=1000):
    response = llm_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    f"{system_prompt} "
                    "Отвечай только валидным JSON без markdown и текста вокруг JSON."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return parse_llm_json(response)


def search_context_by_type(transcript, doc_type, query):
    results = rag.search_knowledge(query=f"{query}\n\n{transcript}", top_k=8)
    filtered = [chunk for chunk in results if chunk.get("doc_type") == doc_type]
    return filtered[:3] if filtered else results[:3]


def retrieve_context_node(state: CallAnalysisState):
    transcript = state["transcript"]
    return {
        "rag_context": {
            "sales_script": search_context_by_type(
                transcript,
                "sales_script",
                "Этапы скрипта продаж и критерии качественного звонка",
            ),
            "objections": search_context_by_type(
                transcript,
                "objections",
                "Типовые возражения клиента и правильная обработка возражений",
            ),
            "product_info": search_context_by_type(
                transcript,
                "product_info",
                "Описание продукта SalesTech CRM Analytics и его польза",
            ),
            "scoring_rubric": search_context_by_type(
                transcript,
                "scoring_rubric",
                "Рубрика оценки sales-звонка на 100 баллов",
            ),
        }
    }


def summary_agent_node(state: CallAnalysisState):
    context = format_chunks(state["rag_context"].get("product_info", []))
    summary = ask_json_agent(
        "Ты агент, который кратко резюмирует звонки отдела продаж.",
        (
            "Верни JSON строго такой структуры:\n"
            '{"summary": "...", "client_need": "...", "next_step": "..."}\n\n'
            f"Контекст:\n{context}\n\n"
            f"Транскрипт:\n{state['transcript']}"
        ),
    )
    return {"summary": summary}


def classifier_agent_node(state: CallAnalysisState):
    classification = ask_json_agent(
        "Ты агент, который классифицирует результат sales-звонка.",
        (
            "Классы результата: successful, follow_up_required, client_refused, "
            "no_decision, bad_call.\n"
            "Верни JSON строго такой структуры:\n"
            '{"call_result": "follow_up_required", "confidence": 0.82, "reason": "..."}\n\n'
            f"Резюме:\n{json.dumps(state['summary'], ensure_ascii=False)}\n\n"
            f"Транскрипт:\n{state['transcript']}"
        ),
    )
    return {"classification": classification}


def script_compliance_agent_node(state: CallAnalysisState):
    context = format_chunks(state["rag_context"].get("sales_script", []))
    script_check = ask_json_agent(
        "Ты агент, который проверяет выполнение этапов скрипта продаж.",
        (
            "Верни JSON строго такой структуры:\n"
            '{"script_steps": [{"step": "Приветствие", "status": "done", "evidence": "..."}]}\n'
            "status может быть done, partial или missed.\n\n"
            f"Скрипт продаж:\n{context}\n\n"
            f"Транскрипт:\n{state['transcript']}"
        ),
        max_tokens=1200,
    )
    return {"script_check": script_check}


def objection_analyzer_agent_node(state: CallAnalysisState):
    context = format_chunks(state["rag_context"].get("objections", []))
    objections = ask_json_agent(
        "Ты агент, который находит и оценивает обработку возражений клиента.",
        (
            "Верни JSON строго такой структуры:\n"
            '{"objections": [{"type": "Цена", "client_phrase": "...", '
            '"manager_response": "...", "quality": "weak", "recommendation": "..."}]}\n'
            "quality может быть strong, medium или weak. Если возражений нет, верни пустой список.\n\n"
            f"База знаний по возражениям:\n{context}\n\n"
            f"Транскрипт:\n{state['transcript']}"
        ),
        max_tokens=1200,
    )
    return {"objections": objections}


def scoring_agent_node(state: CallAnalysisState):
    context = format_chunks(state["rag_context"].get("scoring_rubric", []))
    scoring = ask_json_agent(
        "Ты агент, который оценивает звонок по рубрике на 100 баллов.",
        (
            "Верни JSON строго такой структуры:\n"
            '{"total_score": 72, "scores": {"greeting": 10, "needs_discovery": 14, '
            '"presentation": 16, "objection_handling": 15, "closing": 17}, "grade": "medium"}\n'
            "grade может быть low, medium или high.\n\n"
            f"Рубрика:\n{context}\n\n"
            f"Проверка скрипта:\n{json.dumps(state['script_check'], ensure_ascii=False)}\n\n"
            f"Возражения:\n{json.dumps(state['objections'], ensure_ascii=False)}\n\n"
            f"Транскрипт:\n{state['transcript']}"
        ),
    )
    return {"scoring": scoring}


def coach_agent_node(state: CallAnalysisState):
    recommendations = ask_json_agent(
        "Ты агент-коуч для менеджера по продажам.",
        (
            "Верни JSON строго такой структуры:\n"
            '{"manager_recommendations": ["..."], '
            '"example_better_phrases": [{"situation": "...", "better_phrase": "..."}]}\n\n'
            f"Резюме:\n{json.dumps(state['summary'], ensure_ascii=False)}\n\n"
            f"Классификация:\n{json.dumps(state['classification'], ensure_ascii=False)}\n\n"
            f"Скрипт:\n{json.dumps(state['script_check'], ensure_ascii=False)}\n\n"
            f"Возражения:\n{json.dumps(state['objections'], ensure_ascii=False)}\n\n"
            f"Оценка:\n{json.dumps(state['scoring'], ensure_ascii=False)}"
        ),
        max_tokens=1200,
    )
    return {"recommendations": recommendations}


def report_generator_node(state: CallAnalysisState):
    final_report = {
        "call_id": state["call_id"],
        "summary": state["summary"].get("summary", ""),
        "client_need": state["summary"].get("client_need", ""),
        "next_step": state["summary"].get("next_step", ""),
        "call_result": state["classification"].get("call_result", "no_decision"),
        "classification": state["classification"],
        "total_score": state["scoring"].get("total_score", 0),
        "scoring": state["scoring"],
        "script_compliance": state["script_check"].get("script_steps", []),
        "objections": state["objections"].get("objections", []),
        "recommendations": state["recommendations"].get("manager_recommendations", []),
        "example_better_phrases": state["recommendations"].get("example_better_phrases", []),
    }
    return {"final_report": final_report}


def build_call_analysis_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(CallAnalysisState)
    graph.add_node("retrieve_context_node", retrieve_context_node)
    graph.add_node("summary_agent_node", summary_agent_node)
    graph.add_node("classifier_agent_node", classifier_agent_node)
    graph.add_node("script_compliance_agent_node", script_compliance_agent_node)
    graph.add_node("objection_analyzer_agent_node", objection_analyzer_agent_node)
    graph.add_node("scoring_agent_node", scoring_agent_node)
    graph.add_node("coach_agent_node", coach_agent_node)
    graph.add_node("report_generator_node", report_generator_node)

    graph.add_edge(START, "retrieve_context_node")
    graph.add_edge("retrieve_context_node", "summary_agent_node")
    graph.add_edge("summary_agent_node", "classifier_agent_node")
    graph.add_edge("classifier_agent_node", "script_compliance_agent_node")
    graph.add_edge("script_compliance_agent_node", "objection_analyzer_agent_node")
    graph.add_edge("objection_analyzer_agent_node", "scoring_agent_node")
    graph.add_edge("scoring_agent_node", "coach_agent_node")
    graph.add_edge("coach_agent_node", "report_generator_node")
    graph.add_edge("report_generator_node", END)

    return graph.compile()


def analyze_call_with_agents(call_id, transcript):
    graph = build_call_analysis_graph()
    initial_state = {
        "call_id": call_id,
        "transcript": transcript,
        "rag_context": {},
        "summary": {},
        "classification": {},
        "script_check": {},
        "objections": {},
        "scoring": {},
        "recommendations": {},
        "final_report": {},
    }
    result = graph.invoke(initial_state)
    return result["final_report"]
