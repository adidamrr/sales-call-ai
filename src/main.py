from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
import json
from sqlalchemy.orm import Session

from src import agents, llm_client, models, rag, services
from src.database import Base, engine, get_db
from src.schemas import (
    CallRead,
    CallStatusResponse,
    CallUploadResponse,
    BasicAnalysisResponse,
    LLMChatRequest,
    LLMChatResponse,
    LLMPreviewResponse,
    ManagerCreate,
    ManagerRead,
    RagSearchRequest,
    RagSearchResponse,
    TranscriptCreate,
    TranscriptRead,
    TranscriptUploadResponse,
    TranscribeCallResponse,
)


app = FastAPI(title="Sales Call AI")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/managers", response_model=ManagerRead, status_code=status.HTTP_201_CREATED)
def create_manager(
    manager: ManagerCreate, db: Session = Depends(get_db)
):
    return services.create_manager(db, manager)


@app.get("/managers", response_model=list[ManagerRead])
def get_managers(db: Session = Depends(get_db)):
    return services.get_managers(db)


@app.post("/calls/upload", response_model=CallUploadResponse)
def upload_call(
    manager_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    manager = db.query(models.Manager).filter(models.Manager.id == manager_id).first()
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Manager not found.",
        )

    audio_path = services.save_uploaded_audio(file)
    call = services.create_call(db, manager_id=manager_id, audio_path=audio_path)
    return CallUploadResponse(call_id=call.id, status=call.status)


@app.get("/calls", response_model=list[CallRead])
def get_calls(db: Session = Depends(get_db)):
    return services.get_calls(db)


@app.get("/calls/{call_id}", response_model=CallRead)
def get_call(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )
    return call


@app.get("/calls/{call_id}/status", response_model=CallStatusResponse)
def get_call_status(
    call_id: int, db: Session = Depends(get_db)
):
    call_status = services.get_call_status(db, call_id)
    if call_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )
    return CallStatusResponse(call_id=call_id, status=call_status)


@app.post("/calls/{call_id}/transcript", response_model=TranscriptUploadResponse)
def upload_transcript(
    call_id: int, transcript_data: TranscriptCreate, db: Session = Depends(get_db)
):
    result = services.create_or_update_transcript(
        db, call_id=call_id, text=transcript_data.text
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript, call_status = result
    return TranscriptUploadResponse(
        call_id=call_id,
        transcript_id=transcript.id,
        status=call_status,
    )


@app.get("/calls/{call_id}/transcript", response_model=TranscriptRead)
def get_transcript(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript = services.get_transcript_by_call_id(db, call_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found.",
        )

    return transcript


@app.post("/calls/{call_id}/transcribe", response_model=TranscribeCallResponse)
def transcribe_call(call_id: int, db: Session = Depends(get_db)):
    result = services.create_transcript_from_audio(db, call_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript, call_status = result
    return TranscribeCallResponse(
        call_id=call_id,
        transcript_id=transcript.id,
        status=call_status,
        text=transcript.text,
    )


@app.post("/knowledge/search", response_model=RagSearchResponse)
def search_knowledge(search_request: RagSearchRequest):
    results = rag.search_knowledge(
        query=search_request.query,
        top_k=search_request.top_k,
    )
    return RagSearchResponse(results=results)


@app.post("/llm/chat", response_model=LLMChatResponse)
def llm_chat(chat_request: LLMChatRequest):
    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": chat_request.message}],
        temperature=chat_request.temperature,
        max_tokens=chat_request.max_tokens,
    )
    return LLMChatResponse(response=response)


@app.post("/calls/{call_id}/llm-preview", response_model=LLMPreviewResponse)
def llm_preview(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript = services.get_transcript_by_call_id(db, call_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found.",
        )

    task = (
        "Кратко проанализируй звонок отдела продаж. "
        "Найди основные проблемы и дай 3 рекомендации менеджеру."
    )
    context_chunks = rag.search_knowledge(
        query=f"{task}\n\n{transcript.text}",
        top_k=5,
    )
    response = llm_client.analyze_text_with_context(
        task=task,
        transcript=transcript.text,
        context_chunks=context_chunks,
    )
    return LLMPreviewResponse(call_id=call_id, response=response)


@app.post("/calls/{call_id}/analyze-basic", response_model=BasicAnalysisResponse)
def analyze_basic(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript = services.get_transcript_by_call_id(db, call_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found.",
        )

    services.update_call_status(db, call_id, models.CallStatus.analyzing.value)

    context_chunks = rag.search_knowledge(
        query=transcript.text,
        top_k=5,
    )
    analysis = llm_client.analyze_sales_call_basic(
        transcript=transcript.text,
        context_chunks=context_chunks,
    )
    response = BasicAnalysisResponse(call_id=call_id, **analysis)

    report_data = response.model_dump()
    services.create_or_update_report(
        db,
        call_id=call_id,
        analysis={
            **report_data,
            "report_json": json.dumps(report_data, ensure_ascii=False),
        },
    )
    services.update_call_status(db, call_id, models.CallStatus.completed.value)
    return response


@app.get("/calls/{call_id}/report")
def get_report(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    report = services.get_report_by_call_id(db, call_id)
    if report is None or report.report_json is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    return json.loads(report.report_json)


@app.post("/calls/{call_id}/analyze-agents")
def analyze_agents(call_id: int, db: Session = Depends(get_db)):
    call = services.get_call_by_id(db, call_id)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found.",
        )

    transcript = services.get_transcript_by_call_id(db, call_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found.",
        )

    services.update_call_status(db, call_id, models.CallStatus.analyzing.value)

    final_report = agents.analyze_call_with_agents(
        call_id=call_id,
        transcript=transcript.text,
    )
    services.create_or_update_report(
        db,
        call_id=call_id,
        analysis={
            "summary": final_report.get("summary"),
            "call_result": final_report.get("call_result"),
            "total_score": final_report.get("total_score"),
            "report_json": json.dumps(final_report, ensure_ascii=False),
        },
    )
    services.update_call_status(db, call_id, models.CallStatus.completed.value)
    return final_report
