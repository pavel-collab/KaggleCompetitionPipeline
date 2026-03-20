"""
Main entry point for Kaggle Pipeline.

Usage:
    python -m app.main scheduler   - Run scheduler (fetches competitions periodically)
    python -m app.main classify    - Run classification worker
    python -m app.main notify      - Run notification worker
    python -m app.main init-db     - Initialize database tables
"""

import sys
import time
from datetime import datetime, timedelta

from app.database import (
    count_pending_competitions,
    get_pending_competitions,
    init_db,
)
from app.logging_config import get_logger
from app.telegram import send_error_notification
from app.worker import (
    publish_classify_task,
    publish_notify_task,
    run_classify_worker,
    run_notify_worker,
    wait_for_rabbitmq,
)

logger = get_logger("main")


# Threshold: if we have >= this many pending competitions, skip fetching new ones
PENDING_THRESHOLD = 5

# How many competitions to notify at a time
NOTIFY_BATCH_SIZE = 3

# Scheduler interval (seconds) - every 24 hours
SCHEDULER_INTERVAL = 24 * 60 * 60


# Kaggle API pagination settings
MAX_PAGE_NUMBER = 10
PAGE_SIZE = 100


def fetch_competitions() -> list[dict]:
    """
    Fetch active competitions from Kaggle API.

    Returns competitions that:
    - Have deadline in the future
    - Have at least 7 days until deadline
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    try:
        api = KaggleApi()
        api.authenticate()

        competitions = []
        page = 1
        today_date = datetime.now()

        while True:
            comps = api.competitions_list(
                page=page,
                page_size=PAGE_SIZE,
                search="",
                category="all",
                sort_by="latestDeadline",
            )

            if not comps or not comps.competitions:
                break

            for c in comps.competitions:
                deadline_date = c.deadline

                # Only active competitions
                if deadline_date < today_date:
                    continue

                # At least 7 days until deadline
                if deadline_date - today_date < timedelta(days=7):
                    continue

                competitions.append(c)

            page += 1
            if page >= MAX_PAGE_NUMBER:
                break

        result = []
        for c in competitions:
            description = c.description or ""
            description += "\n\n".join(
                tag.description for tag in c.tags if tag.description
            )

            tags = ", ".join(tag.ref for tag in c.tags if tag.ref)

            result.append(
                {
                    "competition_title": c.title,
                    "link": c.ref,
                    "date_start": c.enabled_date.strftime("%Y-%m-%d"),
                    "deadline": c.deadline.strftime("%Y-%m-%d"),
                    "description": description,
                    "tags": tags,
                }
            )

        return result

    except Exception as e:
        logger.error(f"Failed to fetch competitions from Kaggle API: {e}")
        return []


def run_scheduler() -> None:
    """
    Main scheduler loop.

    Logic (same as n8n workflow):
    1. Check how many pending competitions we have in DB
    2. If >= 5 pending: just send notifications from existing
    3. If < 5 pending: fetch new from Kaggle API
    """
    logger.info("Starting scheduler...")
    init_db()
    wait_for_rabbitmq()

    while True:
        logger.info(f"Running scheduler iteration at {datetime.now()}")

        try:
            pending_count = count_pending_competitions()
            logger.info(f"Pending competitions in DB: {pending_count}")

            if pending_count >= PENDING_THRESHOLD:
                # We have enough pending - just send notifications
                logger.info("Enough pending competitions, sending notifications...")
                competitions = get_pending_competitions(limit=NOTIFY_BATCH_SIZE)

                for comp in competitions:
                    logger.info(f"Publishing notify task for: {comp.title}")
                    publish_notify_task({
                        "title": comp.title,
                        "link": comp.link,
                        "deadline": str(comp.deadline) if comp.deadline else "",
                        "type": comp.type,
                        "description": comp.description or "",
                    })

            else:
                # Need to fetch new competitions
                logger.info("Fetching new competitions from Kaggle API...")

                competitions = fetch_competitions()
                logger.info(f"Fetched {len(competitions)} competitions")

                if not competitions:
                    logger.info("No competitions fetched, skipping...")
                else:
                    # Limit to 50 competitions
                    for comp in competitions[:50]:
                        publish_classify_task(comp)

        except Exception as e:
            logger.error(f"Scheduler error: {type(e).__name__}: {e}")
            send_error_notification(f"Scheduler error: {e}")

        logger.info(f"Sleeping for {SCHEDULER_INTERVAL} seconds...")
        time.sleep(SCHEDULER_INTERVAL)


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    logger.info(f"Starting Kaggle Pipeline with command: {command}")

    if command == "scheduler":
        run_scheduler()
    elif command == "classify":
        init_db()
        run_classify_worker()
    elif command == "notify":
        init_db()
        run_notify_worker()
    elif command == "init-db":
        init_db()
        logger.info("Database initialized!")
    else:
        logger.error(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
