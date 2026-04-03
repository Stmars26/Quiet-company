"""Orchestrator — reads calendar, posts to platforms, logs results."""

import logging
from datetime import datetime
from pathlib import Path

import pytz

from publier.calendar_reader import read_calendar, get_posts_for_date, CET
from publier import db
from publier.platforms import twitter, instagram

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALENDAR_PATH = PROJECT_ROOT / "QC_Master_Social_Calendar.xlsx"
LAUNCH_DIR = PROJECT_ROOT / "static" / "images" / "launch"

PLATFORM_CLIENTS = {
    "X (Twitter)": twitter,
    "Instagram": instagram,
}


def run_scheduled_posts(dry_run: bool = False):
    """Main entry: find today's posts and publish them."""
    db.init_publier_db()

    now = datetime.now(CET)
    logger.info(f"Publier run at {now.strftime('%Y-%m-%d %H:%M %Z')}")

    posts = read_calendar(CALENDAR_PATH, LAUNCH_DIR)
    todays_posts = get_posts_for_date(posts, now)

    if not todays_posts:
        logger.info("No posts scheduled for today.")
        return

    logger.info(f"Found {len(todays_posts)} posts for today")

    posted = 0
    skipped = 0
    failed = 0

    for post in todays_posts:
        platform = post["platform"]
        row = post["row"]

        # Skip if already posted
        if db.is_posted(row, platform):
            logger.info(f"Row {row} [{platform}] already posted, skipping")
            skipped += 1
            continue

        client = PLATFORM_CLIENTS.get(platform)
        if not client:
            logger.warning(f"No client for platform: {platform}")
            continue

        if not client.is_configured():
            logger.warning(f"{platform} not configured (missing API keys), skipping")
            db.log_post(row, platform, "skipped", error="Not configured")
            skipped += 1
            continue

        caption = post["caption"]
        asset_path = post["asset_path"]

        if dry_run:
            has_image = "with image" if asset_path else "text only"
            logger.info(
                f"[DRY RUN] Row {row} [{platform}] {has_image}\n"
                f"  Asset: {post['asset_name']}\n"
                f"  Caption: {caption[:100]}..."
            )
            continue

        # Instagram requires an image
        if platform == "Instagram" and not asset_path:
            logger.warning(f"Row {row} [Instagram] has no image, skipping")
            db.log_post(row, platform, "skipped", caption=caption, error="No image asset")
            skipped += 1
            continue

        try:
            post_id = client.publish(caption, asset_path)
            db.log_post(
                row, platform, "posted",
                caption=caption,
                image_path=str(asset_path) if asset_path else "",
                platform_post_id=post_id,
                scheduled_at=post["date"].isoformat(),
            )
            logger.info(f"Posted row {row} [{platform}] → {post_id}")
            posted += 1
        except Exception as e:
            db.log_post(
                row, platform, "failed",
                caption=caption,
                image_path=str(asset_path) if asset_path else "",
                error=str(e)[:500],
                scheduled_at=post["date"].isoformat(),
            )
            logger.error(f"Failed row {row} [{platform}]: {e}")
            failed += 1

    logger.info(f"Run complete: {posted} posted, {skipped} skipped, {failed} failed")


def retry_failed():
    """Retry posts that previously failed (max 3 attempts)."""
    db.init_publier_db()
    pending = db.get_pending_retries()

    if not pending:
        logger.info("No failed posts to retry")
        return

    logger.info(f"Retrying {len(pending)} failed posts")

    posts = read_calendar(CALENDAR_PATH, LAUNCH_DIR)
    post_by_row = {p["row"]: p for p in posts}

    for entry in pending:
        row = entry["calendar_row"]
        platform = entry["platform"]
        post = post_by_row.get(row)

        if not post:
            logger.warning(f"Row {row} not found in calendar, skipping retry")
            continue

        client = PLATFORM_CLIENTS.get(platform)
        if not client or not client.is_configured():
            continue

        caption = post["caption"]
        asset_path = post["asset_path"]

        if platform == "Instagram" and not asset_path:
            continue

        try:
            post_id = client.publish(caption, asset_path)
            db.log_post(
                row, platform, "posted",
                caption=caption,
                platform_post_id=post_id,
            )
            logger.info(f"Retry success: row {row} [{platform}] → {post_id}")
        except Exception as e:
            db.log_post(row, platform, "failed", caption=caption, error=str(e)[:500])
            logger.error(f"Retry failed: row {row} [{platform}]: {e}")


def show_status():
    """Print the posting log."""
    db.init_publier_db()
    posts = db.get_all_posts()

    if not posts:
        print("No posts logged yet.")
        return

    print(f"\n{'ID':>4}  {'Row':>3}  {'Platform':<14}  {'Status':<8}  {'Retries':>7}  {'Posted At':<20}  {'Error'}")
    print("-" * 95)
    for p in posts:
        print(
            f"{p['id']:>4}  {p['calendar_row']:>3}  {p['platform']:<14}  "
            f"{p['status']:<8}  {p['retries']:>7}  "
            f"{(p['posted_at'] or ''):<20}  {(p['error'] or '')[:40]}"
        )
    print(f"\nTotal: {len(posts)} records")
