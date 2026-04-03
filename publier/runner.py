"""
CLI entry point and scheduler for Publier.

Usage:
    python -m publier dry-run      Preview today's posts without publishing
    python -m publier post         Publish today's posts now
    python -m publier retry        Retry failed posts
    python -m publier status       Show posting log
    python -m publier schedule     Start the scheduler (runs 4x daily at CET posting windows)
"""

import sys
import logging

import schedule
import pytz
from datetime import datetime

from publier.poster import run_scheduled_posts, retry_failed, show_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CET = pytz.timezone("Europe/Amsterdam")


def run_scheduler():
    """
    Run the posting scheduler. Checks 4x daily at optimal CET posting windows:
      12:00 — Instagram midday
      14:00 — Twitter afternoon
      17:00 — Instagram evening
      18:00 — Twitter evening

    Also retries failed posts at 10:00 and 20:00.

    Set TZ=Europe/Amsterdam on Railway so schedule library uses CET.
    """
    logger.info("Publier scheduler starting")

    # Posting windows
    schedule.every().day.at("12:00").do(run_scheduled_posts)
    schedule.every().day.at("14:00").do(run_scheduled_posts)
    schedule.every().day.at("17:00").do(run_scheduled_posts)
    schedule.every().day.at("18:00").do(run_scheduled_posts)

    # Retry windows
    schedule.every().day.at("10:00").do(retry_failed)
    schedule.every().day.at("20:00").do(retry_failed)

    logger.info("Scheduled 4 posting windows + 2 retry windows (CET)")
    logger.info("Next run: " + str(schedule.next_run()))

    import time
    while True:
        schedule.run_pending()
        time.sleep(30)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "dry-run":
        run_scheduled_posts(dry_run=True)
    elif command == "post":
        run_scheduled_posts(dry_run=False)
    elif command == "retry":
        retry_failed()
    elif command == "status":
        show_status()
    elif command == "schedule":
        run_scheduler()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
