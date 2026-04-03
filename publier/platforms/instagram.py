"""
Instagram posting client — Meta Graph API v21.0.

Setup:
  1. Go to developers.facebook.com → create a new App (type: Business)
  2. Add "Instagram Graph API" product
  3. Connect your Instagram Professional Account (Business or Creator)
  4. In Graph API Explorer:
     - Select your app
     - Add permissions: instagram_basic, instagram_content_publish, pages_read_engagement
     - Generate a User Access Token
     - Exchange for a long-lived token (60-day expiry):
       GET https://graph.facebook.com/v21.0/oauth/access_token
         ?grant_type=fb_exchange_token
         &client_id={APP_ID}
         &client_secret={APP_SECRET}
         &fb_exchange_token={SHORT_LIVED_TOKEN}
  5. Get your Instagram User ID:
       GET https://graph.facebook.com/v21.0/me/accounts  (gets Page ID)
       GET https://graph.facebook.com/v21.0/{PAGE_ID}?fields=instagram_business_account
  6. Add to .env:
       INSTAGRAM_ACCESS_TOKEN=...
       INSTAGRAM_USER_ID=...
       QC_BASE_URL=https://your-app.up.railway.app

Note: Images must be publicly accessible URLs.
      Your launch images are served at {QC_BASE_URL}/static/images/launch/{filename}.
      Instagram does NOT accept file uploads — only URLs.
"""

from __future__ import annotations

import os
import logging
import time

import requests

logger = logging.getLogger(__name__)

GRAPH_API = "https://graph.facebook.com/v21.0"


def is_configured() -> bool:
    """Check if Instagram credentials are set."""
    return all(
        os.getenv(k) for k in ["INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID", "QC_BASE_URL"]
    )


def _get_config() -> tuple[str, str, str]:
    return (
        os.environ["INSTAGRAM_ACCESS_TOKEN"],
        os.environ["INSTAGRAM_USER_ID"],
        os.environ["QC_BASE_URL"].rstrip("/"),
    )


def create_container(image_url: str, caption: str) -> str:
    """Create a media container. Returns the container ID."""
    token, ig_user_id, _ = _get_config()

    resp = requests.post(
        f"{GRAPH_API}/{ig_user_id}/media",
        data={
            "image_url": image_url,
            "caption": caption[:2200],
            "access_token": token,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        error = resp.json().get("error", {}).get("message", resp.text)
        raise RuntimeError(f"Instagram container failed: {error}")

    container_id = resp.json()["id"]
    logger.info(f"Created IG container: {container_id}")
    return container_id


def publish_container(container_id: str) -> str:
    """Publish a media container. Returns the media ID."""
    token, ig_user_id, _ = _get_config()

    # Wait for container to be ready (Instagram processes async)
    for attempt in range(5):
        status_resp = requests.get(
            f"{GRAPH_API}/{container_id}",
            params={"fields": "status_code", "access_token": token},
            timeout=15,
        )
        status = status_resp.json().get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            error = status_resp.json().get("status", "Unknown error")
            raise RuntimeError(f"Instagram container error: {error}")
        logger.info(f"Container {container_id} status: {status}, waiting...")
        time.sleep(3)

    resp = requests.post(
        f"{GRAPH_API}/{ig_user_id}/media_publish",
        data={
            "creation_id": container_id,
            "access_token": token,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        error = resp.json().get("error", {}).get("message", resp.text)
        raise RuntimeError(f"Instagram publish failed: {error}")

    media_id = resp.json()["id"]
    logger.info(f"Published to Instagram: {media_id}")
    return media_id


def image_path_to_url(asset_path) -> str | None:
    """Convert a local asset path to a public URL for Instagram."""
    if not asset_path:
        return None
    _, _, base_url = _get_config()
    # Extract the relative path from static/ onwards
    path_str = str(asset_path)
    if "static/" in path_str:
        relative = path_str[path_str.index("static/"):]
        return f"{base_url}/{relative}"
    return None


def publish(caption: str, image_path=None) -> str:
    """
    High-level: create container + publish.
    Returns the Instagram media ID.
    """
    image_url = image_path_to_url(image_path)
    if not image_url:
        raise RuntimeError("Instagram requires an image URL. No valid asset provided.")

    container_id = create_container(image_url, caption)
    media_id = publish_container(container_id)
    return media_id
