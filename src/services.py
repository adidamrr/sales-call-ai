from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from src.config import settings
from src.models import Call, CallStatus, Manager
from src.schemas import ManagerCreate


ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}


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
