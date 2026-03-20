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
from datetime import datetime

import httpx

from app.config import settings
from app.database import (
    count_pending_competitions,
    get_pending_competitions,
    init_db,
)
from app.telegram import send_error_notification
from app.worker import (
    publish_classify_task,
    publish_notify_task,
    run_classify_worker,
    run_notify_worker,
    wait_for_rabbitmq,
)


# Threshold: if we have >= this many pending competitions, skip fetching new ones
PENDING_THRESHOLD = 5

# How many competitions to notify at a time
NOTIFY_BATCH_SIZE = 3

# Scheduler interval (seconds) - every 24 hours
SCHEDULER_INTERVAL = 24 * 60 * 60


def check_kaggle_api_health() -> bool:
    """Check if Kaggle API service is available."""
    try:
        response = httpx.get(f"{settings.kaggle_api_url}/health", timeout=10)
        return response.json().get("status") == "ok"
    except Exception as e:
        print(f"Kaggle API health check failed: {e}")
        return False

#TODO: need to use kaggle api instead of httpx requests
def fetch_competitions() -> list[dict]:
    """Fetch competitions from Kaggle API."""
    try:
        response = httpx.get(f"{settings.kaggle_api_url}/competitions", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except Exception as e:
        print(f"Failed to fetch competitions: {e}")
        return []


def run_scheduler() -> None:
    """
    Main scheduler loop.

    Logic (same as n8n workflow):
    1. Check how many pending competitions we have in DB
    2. If >= 5 pending: just send notifications from existing
    3. If < 5 pending: fetch new from Kaggle API
    """
    print("Starting scheduler...")
    init_db()
    wait_for_rabbitmq()

    while True:
        print(f"\n[{datetime.now()}] Running scheduler iteration...")

        try:
            pending_count = count_pending_competitions()
            print(f"Pending competitions in DB: {pending_count}")

            if pending_count >= PENDING_THRESHOLD:
                # We have enough pending - just send notifications
                print("Enough pending competitions, sending notifications...")
                competitions = get_pending_competitions(limit=NOTIFY_BATCH_SIZE)

                for comp in competitions:
                    publish_notify_task({
                        "title": comp.title,
                        "link": comp.link,
                        "deadline": str(comp.deadline) if comp.deadline else "",
                        "type": comp.type,
                        "description": comp.description or "",
                    })

            else:
                # Need to fetch new competitions
                print("Fetching new competitions from API...")

                if not check_kaggle_api_health():
                    send_error_notification("Kaggle API is not available")
                    print("Kaggle API not available, skipping...")
                else:
                    competitions = fetch_competitions()
                    print(f"Fetched {len(competitions)} competitions")

                    # Limit to 50 competitions
                    for comp in competitions[:50]:
                        publish_classify_task(comp)

        except Exception as e:
            print(f"Scheduler error: {e}")
            send_error_notification(f"Scheduler error: {e}")

        print(f"Sleeping for {SCHEDULER_INTERVAL} seconds...")
        time.sleep(SCHEDULER_INTERVAL)


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

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
        print("Database initialized!")
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
