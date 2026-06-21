from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from src import models, rag, services
from src.database import Base, engine, get_db
from src.schemas import (
    CallRead,
    CallStatusResponse,
    CallUploadResponse,
    ManagerCreate,
    ManagerRead,
    RagSearchRequest,
    RagSearchResponse,
    TranscriptCreate,
    TranscriptRead,
    TranscriptUploadResponse,
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


@app.post("/knowledge/search", response_model=RagSearchResponse)
def search_knowledge(search_request: RagSearchRequest):
    results = rag.search_knowledge(
        query=search_request.query,
        top_k=search_request.top_k,
    )
    return RagSearchResponse(results=results)
