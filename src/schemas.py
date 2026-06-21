from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ManagerCreate(BaseModel):
    name: str
    department: str


class ManagerRead(BaseModel):
    id: int
    name: str
    department: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CallRead(BaseModel):
    id: int
    manager_id: int
    audio_path: str
    transcript_path: str | None
    status: str
    duration_seconds: int | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CallUploadResponse(BaseModel):
    call_id: int
    status: str


class CallStatusResponse(BaseModel):
    call_id: int
    status: str


class TranscriptRead(BaseModel):
    id: int
    call_id: int
    text: str
    segments_json: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportRead(BaseModel):
    id: int
    call_id: int
    summary: str | None
    call_result: str | None
    total_score: int | None
    report_json: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
