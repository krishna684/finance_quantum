from __future__ import annotations

import time
from typing import Any

import requests


class TikTokResearchAPI:
    BASE_URL = "https://open.tiktokapis.com"

    def __init__(self, client_key: str, client_secret: str, access_token: str = "") -> None:
        self.client_key = client_key
        self.client_secret = client_secret
        self.access_token = access_token

    def authenticate(self) -> str:
        if self.access_token:
            return self.access_token

        url = f"{self.BASE_URL}/v2/oauth/token/"
        payload = {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"No access_token returned: {data}")
        self.access_token = token
        return token

    def _post(self, path: str, fields: list[str], body: dict[str, Any]) -> dict[str, Any]:
        token = self.authenticate()
        url = f"{self.BASE_URL}{path}"
        params = {"fields": ",".join(fields)}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        retries = 4
        for attempt in range(retries + 1):
            response = requests.post(url, params=params, json=body, headers=headers, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            err = (payload or {}).get("error", {})
            if err and err.get("code") not in ("ok", 0, None):
                raise RuntimeError(f"TikTok API error: {err}")
            return payload.get("data", {})
        raise RuntimeError("TikTok API retry limit exceeded")

    def query_videos(
        self,
        query: dict[str, Any],
        start_date: str,
        end_date: str,
        cursor: int = 0,
        max_count: int = 100,
        is_random: bool = False,
        search_id: str | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        video_fields = fields or [
            "id",
            "create_time",
            "username",
            "video_description",
            "hashtag_names",
            "voice_to_text",
            "view_count",
            "like_count",
            "share_count",
            "comment_count",
            "region_code",
        ]

        body: dict[str, Any] = {
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "cursor": cursor,
            "max_count": max_count,
            "is_random": is_random,
        }
        if search_id:
            body["search_id"] = search_id

        return self._post(
            "/v2/research/video/query/",
            fields=video_fields,
            body=body,
        )

    def query_video_comments(
        self,
        video_id: int,
        cursor: int = 0,
        max_count: int = 100,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        comment_fields = fields or [
            "id",
            "video_id",
            "text",
            "like_count",
            "reply_count",
            "parent_comment_id",
            "create_time",
        ]
        body = {
            "video_id": int(video_id),
            "cursor": cursor,
            "max_count": max_count,
        }
        return self._post(
            "/v2/research/video/comment/list/",
            fields=comment_fields,
            body=body,
        )
