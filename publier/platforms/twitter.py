"""
X/Twitter posting client — v2 API with OAuth 1.0a.

Setup:
  1. Go to developer.x.com → sign up for Free tier
  2. Create a Project + App
  3. Enable OAuth 1.0a with Read + Write permissions
  4. Generate: API Key, API Secret, Access Token, Access Token Secret
  5. Add to .env:
       TWITTER_API_KEY=...
       TWITTER_API_SECRET=...
       TWITTER_ACCESS_TOKEN=...
       TWITTER_ACCESS_SECRET=...

Free tier limits: 1,500 tweets/month, 1 app.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import requests
from requests_oauthlib import OAuth1

logger = logging.getLogger(__name__)

UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
TWEET_URL = "https://api.twitter.com/2/tweets"


def get_auth() -> OAuth1:
    """Build OAuth1 credentials from env vars."""
    return OAuth1(
        os.environ["TWITTER_API_KEY"],
        os.environ["TWITTER_API_SECRET"],
        os.environ["TWITTER_ACCESS_TOKEN"],
        os.environ["TWITTER_ACCESS_SECRET"],
    )


def is_configured() -> bool:
    """Check if Twitter credentials are set."""
    keys = ["TWITTER_API_KEY", "TWITTER_API_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"]
    for k in keys:
        v = os.getenv(k, "")
        logger.info(f"  {k}: {'set (' + v[:4] + '...' + v[-4:] + ')' if len(v) > 8 else 'MISSING' if not v else 'set (short)'}")
    return all(os.getenv(k) for k in keys)


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
    logger.info(f"Uploaded media: {image_path.name} → {media_id}")
    return media_id


def post_tweet(text: str, media_ids: list[str] | None = None) -> dict:
    """
    Post a tweet. Returns {"id": "...", "text": "..."} on success.
    Raises on failure.
    """
    auth = get_auth()

    payload = {"text": text[:280]}
    if media_ids:
        payload["media"] = {"media_ids": media_ids}

    resp = requests.post(
        TWEET_URL,
        auth=auth,
        json=payload,
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        error = resp.text
        logger.error(f"Tweet failed: {resp.status_code} {error}")
        raise RuntimeError(f"Tweet failed ({resp.status_code}): {error}")

    data = resp.json().get("data", {})
    logger.info(f"Tweet posted: {data.get('id')}")
    return data


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
