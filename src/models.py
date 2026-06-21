from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class CallStatus(str, Enum):
    uploaded = "uploaded"
    transcribing = "transcribing"
    transcribed = "transcribed"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"


class Manager(Base):
    __tablename__ = "managers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    calls: Mapped[list["Call"]] = relationship(
        back_populates="manager", cascade="all, delete-orphan"
    )


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    manager_id: Mapped[int] = mapped_column(ForeignKey("managers.id"), nullable=False)
    audio_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    transcript_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default=CallStatus.uploaded.value, nullable=False
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    manager: Mapped["Manager"] = relationship(back_populates="calls")
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="call", cascade="all, delete-orphan", uselist=False
    )
    report: Mapped["Report | None"] = relationship(
        back_populates="call", cascade="all, delete-orphan", uselist=False
    )


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("calls.id"), nullable=False, unique=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    segments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    call: Mapped["Call"] = relationship(back_populates="transcript")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    call_id: Mapped[int] = mapped_column(
        ForeignKey("calls.id"), nullable=False, unique=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    call_result: Mapped[str | None] = mapped_column(String(255), nullable=True)
    total_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    call: Mapped["Call"] = relationship(back_populates="report")
