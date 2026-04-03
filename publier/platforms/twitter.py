"""
X/Twitter posting client — tries v2 API first, falls back to v1.1.

Pay-per-use standalone apps may not have v2 POST access (known X bug).
The v1.1 fallback uses /1.1/statuses/update.json which works with
standalone apps + OAuth 1.0a.

Setup:
  1. Go to developer.x.com → sign up (pay-per-use)
  2. Create an App with Read + Write permissions
  3. Set up User Authentication (Web App type) with callback URL
  4. Generate: Consumer Key, Consumer Secret, Access Token, Access Token Secret
  5. Add to .env:
       TWITTER_API_KEY=...       (Consumer Key)
       TWITTER_API_SECRET=...    (Consumer Secret)
       TWITTER_ACCESS_TOKEN=...
       TWITTER_ACCESS_SECRET=...
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import requests
from requests_oauthlib import OAuth1

logger = logging.getLogger(__name__)

UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
TWEET_V2_URL = "https://api.twitter.com/2/tweets"
TWEET_V1_URL = "https://api.twitter.com/1.1/statuses/update.json"


def get_auth() -> OAuth1:
    """Build OAuth1 credentials from env vars."""
    return OAuth1(
        os.environ["TWITTER_API_KEY"].strip(),
        os.environ["TWITTER_API_SECRET"].strip(),
        os.environ["TWITTER_ACCESS_TOKEN"].strip(),
        os.environ["TWITTER_ACCESS_SECRET"].strip(),
    )


def is_configured() -> bool:
    """Check if Twitter credentials are set."""
    keys = ["TWITTER_API_KEY", "TWITTER_API_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"]
    for k in keys:
        v = os.getenv(k, "").strip()
        if len(v) > 8:
            logger.info(f"  {k}: set ({v[:4]}...{v[-4:]}, len={len(v)})")
        elif not v:
            logger.info(f"  {k}: MISSING")
        else:
            logger.info(f"  {k}: set (short, len={len(v)})")
    return all(os.getenv(k, "").strip() for k in keys)


def upload_media(image_path: Path) -> str | None:
    """Upload an image and return the media_id string."""
    if not image_path or not image_path.exists():
        return None

    auth = get_auth()
    with open(image_path, "rb") as f:
        resp = requests.post(
            UPLOAD_URL,
            auth=auth,
            files={"media": f},
            timeout=60,
        )

    if resp.status_code != 200:
        logger.error(f"Twitter media upload failed: {resp.status_code} {resp.text}")
        return None

    media_id = resp.json().get("media_id_string")
    logger.info(f"Uploaded media: {image_path.name} -> {media_id}")
    return media_id


def post_tweet_v2(text: str, auth: OAuth1, media_ids: list[str] | None = None) -> dict | None:
    """Try posting via v2 API. Returns result dict or None on auth failure."""
    payload = {"text": text[:280]}
    if media_ids:
        payload["media"] = {"media_ids": media_ids}

    logger.info(f"Trying v2 API: {TWEET_V2_URL}")
    resp = requests.post(
        TWEET_V2_URL,
        auth=auth,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if resp.status_code in (200, 201):
        data = resp.json().get("data", {})
        logger.info(f"Tweet posted via v2: {data.get('id')}")
        return data

    if resp.status_code in (401, 403):
        logger.warning(f"v2 API returned {resp.status_code}, will try v1.1 fallback")
        return None

    # Other error — raise
    raise RuntimeError(f"Tweet v2 failed ({resp.status_code}): {resp.text}")


def post_tweet_v1(text: str, auth: OAuth1, media_ids: list[str] | None = None) -> dict:
    """Post via v1.1 API (works with standalone apps)."""
    params = {"status": text[:280]}
    if media_ids:
        params["media_ids"] = ",".join(media_ids)

    logger.info(f"Trying v1.1 API: {TWEET_V1_URL}")
    resp = requests.post(
        TWEET_V1_URL,
        auth=auth,
        data=params,
        timeout=30,
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        tweet_id = str(data.get("id_str", data.get("id", "")))
        logger.info(f"Tweet posted via v1.1: {tweet_id}")
        return {"id": tweet_id, "text": data.get("text", "")}

    error = resp.text
    logger.error(f"Tweet v1.1 failed: {resp.status_code} {error}")
    raise RuntimeError(f"Tweet failed ({resp.status_code}): {error}")


def post_tweet(text: str, media_ids: list[str] | None = None) -> dict:
    """
    Post a tweet. Tries v2 first, falls back to v1.1.
    Returns {"id": "...", "text": "..."} on success.
    """
    auth = get_auth()

    # Try v2 first
    result = post_tweet_v2(text, auth, media_ids)
    if result:
        return result

    # Fall back to v1.1
    logger.info("Falling back to v1.1 API (standalone app / pay-per-use)")
    return post_tweet_v1(text, auth, media_ids)


def publish(text: str, image_path: Path | None = None) -> str:
    """
    High-level: upload image (if any) + post tweet.
    Returns the tweet ID.
    """
    media_ids = []
    if image_path:
        media_id = upload_media(image_path)
        if media_id:
            media_ids.append(media_id)

    result = post_tweet(text, media_ids or None)
    return result.get("id", "")
