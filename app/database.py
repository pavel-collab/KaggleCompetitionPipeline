"""
Database connection and CRUD operations for Kaggle competitions.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base, CompetitionStatus, KaggleCompetition

# Create engine and session factory
engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables in database."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get new database session."""
    return SessionLocal()


def count_pending_competitions() -> int:
    """Count competitions with status 'new' or 'queued'."""
    with get_session() as session:
        result = session.execute(
            select(KaggleCompetition).where(
                or_(
                    KaggleCompetition.status == CompetitionStatus.NEW,
                    KaggleCompetition.status == CompetitionStatus.QUEUED,
                )
            )
        )
        return len(result.scalars().all())


def upsert_competition(
    title: str,
    link: str,
    date_start: Optional[datetime],
    deadline: Optional[datetime],
    description: Optional[str],
    competition_type: Optional[str],
) -> KaggleCompetition:
    """Insert or update competition by title."""
    with get_session() as session:
        # Try to find existing
        competition = session.get(KaggleCompetition, title)

        if competition:
            # Update existing (don't change status)
            competition.link = link
            competition.date_start = date_start
            competition.deadline = deadline
            competition.description = description
            competition.type = competition_type
        else:
            # Create new
            competition = KaggleCompetition(
                title=title,
                link=link,
                date_start=date_start,
                deadline=deadline,
                description=description,
                type=competition_type,
                status=CompetitionStatus.NEW,
            )
            session.add(competition)

        session.commit()
        session.refresh(competition)
        return competition


def get_pending_competitions(limit: int = 3) -> list[KaggleCompetition]:
    """Get competitions with status 'new' or 'queued'."""
    with get_session() as session:
        result = session.execute(
            select(KaggleCompetition)
            .where(
                or_(
                    KaggleCompetition.status == CompetitionStatus.NEW,
                    KaggleCompetition.status == CompetitionStatus.QUEUED,
                )
            )
            .limit(limit)
        )
        return list(result.scalars().all())


def mark_as_shown(title: str) -> None:
    """Mark competition as shown."""
    with get_session() as session:
        competition = session.get(KaggleCompetition, title)
        if competition:
            competition.status = CompetitionStatus.SHOWN
            session.commit()
