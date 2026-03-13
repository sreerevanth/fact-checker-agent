"""
News Scraper & RSS Feed Ingester
=================================
Fetches live articles from RSS feeds and news APIs, then pushes them
into the RAG retriever for real-time knowledge ingestion.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import aiohttp
import feedparser
import yaml

from src.utils.logging import get_logger
from src.utils.models import FeedItem, SourceCategory
from src.utils.settings import get_settings

log = get_logger(__name__)


class SourceRegistry:
    """Loads trusted source config and provides lookup utilities."""

    def __init__(self, config_path: str) -> None:
        import pathlib
        p = pathlib.Path(config_path)
        if not p.exists():
            self._sources: list[dict] = []
            self._low_credibility: set[str] = set()
            log.warning("Trusted sources config not found", path=str(p))
            return

        with open(p) as f:
            data = yaml.safe_load(f)

        self._sources = []
        for section in data.values():
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict) and "name" in item:
                        self._sources.append(item)

        self._low_credibility: set[str] = {
            d for d in (data.get("low_credibility_domains") or [])
        }

    def get_credibility(self, url: str) -> tuple[float, str]:
        """Return (credibility_score, source_name) for a URL domain."""
        if not url:
            return 0.5, "Unknown"

        domain = urlparse(url).netloc.lower().replace("www.", "")

        if domain in self._low_credibility:
            return 0.1, domain

        for src in self._sources:
            src_domain = urlparse(src.get("url", "")).netloc.lower().replace("www.", "")
            if src_domain and src_domain in domain:
                return src.get("credibility", 0.5), src["name"]

        return 0.5, domain

    @property
    def rss_feeds(self) -> list[dict]:
        return [s for s in self._sources if s.get("feed")]


class RSSIngester:
    """Polls RSS feeds and returns FeedItems."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry
        self._seen: set[str] = set()

    async def fetch_feed(
        self,
        session: aiohttp.ClientSession,
        feed_url: str,
        source_name: str,
        credibility: float,
        category: str,
    ) -> list[dict[str, Any]]:
        try:
            async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
        except Exception as exc:
            log.warning("Feed fetch failed", feed=feed_url, error=str(exc))
            return []

        parsed = feedparser.parse(text)
        docs: list[dict[str, Any]] = []

        for entry in parsed.entries[:20]:  # cap per feed
            url = entry.get("link", "")
            uid = hashlib.md5(url.encode()).hexdigest()
            if uid in self._seen:
                continue
            self._seen.add(uid)

            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            # Clean HTML tags from summary
            if "<" in summary:
                from html.parser import HTMLParser
                class _Strip(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.parts: list[str] = []
                    def handle_data(self, d): self.parts.append(d)
                p = _Strip(); p.feed(summary)
                summary = " ".join(p.parts)

            text = f"{title}. {summary}".strip()
            if not text or len(text) < 30:
                continue

            published_raw = entry.get("published_parsed")
            published_at = None
            if published_raw:
                try:
                    published_at = datetime(*published_raw[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass

            docs.append({
                "id": uid,
                "text": text[:2000],  # truncate
                "url": url,
                "source_name": source_name,
                "credibility_score": credibility,
                "category": category,
                "published_at": published_at,
            })

        return docs

    async def fetch_all(self) -> list[dict[str, Any]]:
        """Fetch all registered RSS feeds concurrently."""
        settings = get_settings()
        feeds = self._registry.rss_feeds
        if not feeds:
            log.warning("No RSS feeds configured")
            return []

        connector = aiohttp.TCPConnector(limit=settings.scraper_concurrency)
        headers = {"User-Agent": "FactCheckBot/1.0 (research; +https://github.com/fact-checker)"}
        all_docs: list[dict[str, Any]] = []

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [
                self.fetch_feed(
                    session=session,
                    feed_url=src["feed"],
                    source_name=src["name"],
                    credibility=src.get("credibility", 0.5),
                    category=src.get("category", "unknown"),
                )
                for src in feeds
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_docs.extend(res)

        log.info("RSS ingestion complete", new_docs=len(all_docs))
        return all_docs


class NewsAPIClient:
    """Fetches articles from NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str, registry: SourceRegistry) -> None:
        self._key = api_key
        self._registry = registry

    async def search(self, query: str, page_size: int = 10) -> list[dict[str, Any]]:
        if not self._key:
            log.debug("NewsAPI key not configured — skipping")
            return []

        params = {
            "q": query,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self._key,
            "language": "en",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/everything",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
        except Exception as exc:
            log.warning("NewsAPI request failed", error=str(exc))
            return []

        docs: list[dict[str, Any]] = []
        for article in data.get("articles", []):
            url = article.get("url", "")
            credibility, source_name = self._registry.get_credibility(url)
            title = article.get("title", "")
            description = article.get("description") or ""
            content = article.get("content") or ""
            text = f"{title}. {description} {content}".strip()[:2000]

            if not text:
                continue

            uid = hashlib.md5(url.encode()).hexdigest()
            docs.append({
                "id": uid,
                "text": text,
                "url": url,
                "source_name": source_name,
                "credibility_score": credibility,
                "category": "unknown",
                "published_at": article.get("publishedAt"),
            })

        return docs


class LiveFeedMonitor:
    """
    Continuously polls news feeds and ingests new content into the RAG index.
    Designed to run as a background asyncio task.
    """

    def __init__(self, retriever: Any, registry: SourceRegistry) -> None:
        self._retriever = retriever
        self._rss = RSSIngester(registry)
        self._running = False
        self._poll_interval = 300  # 5 minutes

    async def start(self) -> None:
        self._running = True
        log.info("Live feed monitor started", interval_s=self._poll_interval)
        while self._running:
            try:
                docs = await self._rss.fetch_all()
                if docs:
                    added = await self._retriever.add_documents(docs)
                    log.info("Ingested new documents", count=added)
            except Exception as exc:
                log.error("Feed monitor error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False
        log.info("Live feed monitor stopped")
