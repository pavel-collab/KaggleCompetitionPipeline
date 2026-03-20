"""
SQLAlchemy models for Kaggle competitions.
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class CompetitionStatus(str, PyEnum):
    """Status of competition processing."""
    NEW = "new"
    QUEUED = "queued"
    SHOWN = "shown"


class CompetitionType(str, PyEnum):
    """Type of ML competition."""
    CLASSIC_ML = "CLASSIC ML"
    LLM_NLP = "LLM/NLP"
    CV = "CV"
    OTHER = "Other"


class KaggleCompetition(Base):
    """Kaggle competition model."""

    __tablename__ = "kaggle_competitions"

    # Primary key is title (as in original n8n workflow)
    title: Mapped[str] = mapped_column(String(500), primary_key=True)
    link: Mapped[str] = mapped_column(String(1000), nullable=False)
    date_start: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    deadline: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(CompetitionStatus),
        default=CompetitionStatus.NEW,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<KaggleCompetition(title={self.title!r}, status={self.status})>"
