#!/usr/bin/env python3
"""
CLI entry point for the Fact-Checking Agent.

Usage:
  python -m src.main serve        # Start the API server
  python -m src.main check "..."  # Verify a single claim from CLI
  python -m src.main ingest       # Force a feed ingestion cycle
"""
from __future__ import annotations

import asyncio
import sys


def serve() -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    from src.utils.settings import get_settings

    s = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=s.api_host,
        port=s.api_port,
        workers=s.api_workers if not s.api_reload else 1,
        reload=s.api_reload,
        log_level=s.log_level.lower(),
    )


async def _check_cli(claim_text: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from src.agent.fact_checker import FactCheckAgent
    from src.rag.retriever import FAISSRetriever
    from src.scrapers.news_scraper import RSSIngester, SourceRegistry
    from src.utils.models import Claim
    from src.utils.settings import get_settings

    console = Console()
    settings = get_settings()

    console.print("\n[bold cyan]🔍 Real-Time Fact-Checking Agent[/bold cyan]\n")
    console.print(f"[dim]Claim:[/dim] {claim_text}\n")

    with console.status("[bold green]Initializing retriever..."):
        retriever = FAISSRetriever()
        await retriever.initialize()

    registry = SourceRegistry(str(settings.trusted_sources_config))

    with console.status("[bold green]Fetching live news feeds..."):
        ingester = RSSIngester(registry)
        docs = await ingester.fetch_all()
        if docs:
            await retriever.add_documents(docs)
            console.print(f"[dim]Indexed {len(docs)} documents from {len(registry.rss_feeds)} feeds[/dim]")

    agent = FactCheckAgent(retriever, registry)
    claim = Claim(text=claim_text)

    with console.status("[bold green]Verifying claim..."):
        result = await agent.verify(claim)

    # Display result
    level_colors = {
        "VERIFIED": "green",
        "LIKELY_TRUE": "green",
        "UNVERIFIED": "yellow",
        "MISLEADING": "orange3",
        "LIKELY_FALSE": "red",
        "FALSE": "red",
        "SATIRE": "blue",
        "OUTDATED": "yellow",
    }
    color = level_colors.get(result.credibility_level.value, "white")

    console.print(Panel(
        f"[bold {color}]{result.credibility_level.value}[/bold {color}]\n\n"
        f"[bold]Confidence:[/bold] {result.confidence_score:.0%}\n\n"
        f"[bold]Summary:[/bold]\n{result.summary}\n\n"
        f"[bold]Analysis:[/bold]\n{result.detailed_analysis}",
        title="Verification Result",
        border_style=color,
    ))

    if result.warnings:
        console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
        for w in result.warnings:
            console.print(f"  • {w}")

    # Evidence table
    all_ev = result.supporting_evidence + result.contradicting_evidence
    if all_ev:
        table = Table(title="Evidence Used", show_lines=True)
        table.add_column("Source", style="cyan", max_width=25)
        table.add_column("Credibility", justify="center", max_width=10)
        table.add_column("Classification", justify="center", max_width=14)
        table.add_column("Excerpt", max_width=60)
        for ev in all_ev[:8]:
            cls_color = "green" if ev.supports_claim else "red"
            cls_label = "SUPPORTS" if ev.supports_claim else "CONTRADICTS"
            table.add_row(
                ev.source.name,
                f"{ev.source.credibility_score:.2f}",
                f"[{cls_color}]{cls_label}[/{cls_color}]",
                ev.text[:200],
            )
        console.print(table)

    console.print(f"\n[dim]Latency: {result.latency_ms:.0f}ms | Sources checked: {result.sources_checked}[/dim]\n")


async def _ingest() -> None:
    from src.rag.retriever import FAISSRetriever
    from src.scrapers.news_scraper import RSSIngester, SourceRegistry
    from src.utils.settings import get_settings

    settings = get_settings()
    retriever = FAISSRetriever()
    await retriever.initialize()
    registry = SourceRegistry(str(settings.trusted_sources_config))
    ingester = RSSIngester(registry)
    docs = await ingester.fetch_all()
    added = await retriever.add_documents(docs)
    await retriever.save()
    print(f"Ingested and indexed {added} documents.")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "serve":
        serve()
    elif args[0] == "check":
        if len(args) < 2:
            print("Usage: python -m src.main check '<claim text>'")
            sys.exit(1)
        asyncio.run(_check_cli(" ".join(args[1:])))
    elif args[0] == "ingest":
        asyncio.run(_ingest())
    else:
        print(f"Unknown command: {args[0]}")
        print("Available commands: serve, check, ingest")
        sys.exit(1)


if __name__ == "__main__":
    main()
