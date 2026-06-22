from pathlib import Path
import json
import logging
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from src import agents, llm_client, rag
from src.config import settings
from src.database import SessionLocal
from src.models import Call, CallStatus, Manager, Report, Transcript
from src.schemas import BasicAnalysisResponse, ManagerCreate
from src.stt import transcribe_audio


ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
logger = logging.getLogger(__name__)


def create_manager(db: Session, manager: ManagerCreate):
    db_manager = Manager(name=manager.name, department=manager.department)
    db.add(db_manager)
    db.commit()
    db.refresh(db_manager)
    return db_manager


def get_managers(db: Session):
    return db.query(Manager).order_by(Manager.id).all()


def save_uploaded_audio(file: UploadFile):
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio format. Use .mp3, .wav, or .m4a.",
        )

    audio_dir = Path(settings.AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{uuid4().hex}{extension}"
    file_path = audio_dir / file_name

    with file_path.open("wb") as destination:
        while chunk := file.file.read(1024 * 1024):
            destination.write(chunk)

    return str(file_path)


def create_call(db: Session, manager_id: int, audio_path: str):
    db_call = Call(
        manager_id=manager_id,
        audio_path=audio_path,
        status=CallStatus.uploaded.value,
    )
    db.add(db_call)
    db.commit()
    db.refresh(db_call)
    return db_call


def get_calls(db: Session):
    return db.query(Call).order_by(Call.id).all()


def get_call_by_id(db: Session, call_id: int):
    return db.query(Call).filter(Call.id == call_id).first()


def get_call_status(db: Session, call_id: int):
    call = get_call_by_id(db, call_id)
    if call is None:
        return None
    return call.status


def save_transcript_text(call_id: int, text: str):
    transcripts_dir = Path(settings.TRANSCRIPTS_DIR)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    file_path = transcripts_dir / f"call_{call_id}_transcript.txt"
    file_path.write_text(text, encoding="utf-8")
    return str(file_path)


def get_transcript_by_call_id(db: Session, call_id: int):
    return db.query(Transcript).filter(Transcript.call_id == call_id).first()


def create_transcript_from_audio(db: Session, call_id: int):
    call = get_call_by_id(db, call_id)
    if call is None:
        return None

    audio_path = Path(call.audio_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found.",
        )

    call.status = CallStatus.transcribing.value
    db.commit()
    db.refresh(call)

    try:
        transcription = transcribe_audio(str(audio_path))
        transcript_path = save_transcript_text(call_id, transcription["text"])
        transcript = db.query(Transcript).filter(Transcript.call_id == call_id).first()

        if transcript is None:
            transcript = Transcript(call_id=call_id, text=transcription["text"])
            db.add(transcript)
        else:
            transcript.text = transcription["text"]

        transcript.segments_json = json.dumps(
            transcription["segments"],
            ensure_ascii=False,
        )
        call.transcript_path = transcript_path
        call.status = CallStatus.transcribed.value

        db.commit()
        db.refresh(transcript)
        db.refresh(call)
        return transcript, call.status
    except Exception:
        call.status = CallStatus.failed.value
        db.commit()
        raise


def update_call_status(db: Session, call_id: int, status: str):
    call = get_call_by_id(db, call_id)
    if call is None:
        return None

    call.status = status
    db.commit()
    db.refresh(call)
    return call


def create_or_update_report(db: Session, call_id: int, analysis: dict):
    report_json = analysis.get("report_json")
    report = db.query(Report).filter(Report.call_id == call_id).first()

    if report is None:
        report = Report(call_id=call_id)
        db.add(report)

    report.summary = analysis.get("summary")
    report.call_result = analysis.get("call_result")
    report.total_score = analysis.get("total_score")
    report.report_json = report_json

    db.commit()
    db.refresh(report)
    return report


def get_report_by_call_id(db: Session, call_id: int):
    return db.query(Report).filter(Report.call_id == call_id).first()


def save_analysis_error(db: Session, call_id: int, error: Exception):
    error_report = {
        "call_id": call_id,
        "status": CallStatus.failed.value,
        "error": str(error),
    }
    create_or_update_report(
        db,
        call_id=call_id,
        analysis={
            "summary": "Analysis failed.",
            "call_result": CallStatus.failed.value,
            "total_score": 0,
            "report_json": json.dumps(error_report, ensure_ascii=False),
        },
    )


def run_transcription_task(call_id: int):
    db = SessionLocal()
    try:
        create_transcript_from_audio(db, call_id)
    except Exception:
        logger.exception("Transcription task failed for call_id=%s", call_id)
        db.rollback()
        update_call_status(db, call_id, CallStatus.failed.value)
    finally:
        db.close()


def run_basic_analysis_task(call_id: int):
    db = SessionLocal()
    try:
        transcript = get_transcript_by_call_id(db, call_id)
        if transcript is None:
            raise ValueError("Transcript not found.")

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

        create_or_update_report(
            db,
            call_id=call_id,
            analysis={
                **report_data,
                "report_json": json.dumps(report_data, ensure_ascii=False),
            },
        )
        update_call_status(db, call_id, CallStatus.completed.value)
    except Exception as error:
        logger.exception("Basic analysis task failed for call_id=%s", call_id)
        db.rollback()
        update_call_status(db, call_id, CallStatus.failed.value)
        save_analysis_error(db, call_id, error)
    finally:
        db.close()


def run_agent_analysis_task(call_id: int):
    db = SessionLocal()
    try:
        transcript = get_transcript_by_call_id(db, call_id)
        if transcript is None:
            raise ValueError("Transcript not found.")

        final_report = agents.analyze_call_with_agents(
            call_id=call_id,
            transcript=transcript.text,
        )
        create_or_update_report(
            db,
            call_id=call_id,
            analysis={
                "summary": final_report.get("summary"),
                "call_result": final_report.get("call_result"),
                "total_score": final_report.get("total_score"),
                "report_json": json.dumps(final_report, ensure_ascii=False),
            },
        )
        update_call_status(db, call_id, CallStatus.completed.value)
    except Exception as error:
        logger.exception("Agent analysis task failed for call_id=%s", call_id)
        db.rollback()
        update_call_status(db, call_id, CallStatus.failed.value)
        save_analysis_error(db, call_id, error)
    finally:
        db.close()
