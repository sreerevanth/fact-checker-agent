#!/usr/bin/env python3
"""
Demo script — sends sample claims to the running API and pretty-prints results.
Usage: python scripts/demo.py
Requires the API server to be running: python -m src.main serve
"""
import asyncio
import json

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

BASE_URL = "http://localhost:8000"

SAMPLE_CLAIMS = [
    {
        "claim": "NASA confirmed that drinking bleach cures COVID-19.",
        "label": "Dangerous misinformation",
    },
    {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "label": "Common misconception",
    },
    {
        "claim": "Albert Einstein failed mathematics as a child.",
        "label": "Popular myth",
    },
    {
        "claim": "Climate scientists have reached a broad consensus that human activities are causing global warming.",
        "label": "Scientific consensus",
    },
    {
        "claim": "Vaccines cause autism according to a 1998 study.",
        "label": "Debunked study",
    },
]

LEVEL_COLORS = {
    "VERIFIED": "green",
    "LIKELY_TRUE": "green",
    "UNVERIFIED": "yellow",
    "MISLEADING": "orange3",
    "LIKELY_FALSE": "red",
    "FALSE": "red",
    "SATIRE": "blue",
    "OUTDATED": "yellow",
}

LEVEL_ICONS = {
    "VERIFIED": "✅",
    "LIKELY_TRUE": "🟢",
    "UNVERIFIED": "🟡",
    "MISLEADING": "🟠",
    "LIKELY_FALSE": "🔴",
    "FALSE": "❌",
    "SATIRE": "🎭",
    "OUTDATED": "⏰",
}


async def check_health(client: httpx.AsyncClient, console: Console) -> bool:
    try:
        resp = await client.get(f"{BASE_URL}/health")
        data = resp.json()
        console.print(f"[green]✓ API online[/green] | model={data['model']} | index_size={data['index_size']}\n")
        return True
    except Exception as e:
        console.print(f"[red]✗ API not reachable at {BASE_URL}: {e}[/red]")
        console.print("[yellow]Start the server first: python -m src.main serve[/yellow]")
        return False


async def verify_claim(client: httpx.AsyncClient, claim_text: str, fast: bool = False) -> dict:
    resp = await client.post(
        f"{BASE_URL}/verify",
        json={"claim": claim_text, "fast_mode": fast},
        timeout=60.0,
    )
    return resp.json()


async def main() -> None:
    console = Console()
    console.print("\n[bold cyan]🔍 Fact-Checking Agent — Demo[/bold cyan]\n")

    async with httpx.AsyncClient() as client:
        if not await check_health(client, console):
            return

        table = Table(title="Batch Verification Results", show_lines=True, expand=True)
        table.add_column("Label", style="dim", max_width=22)
        table.add_column("Verdict", justify="center", max_width=14)
        table.add_column("Conf.", justify="center", max_width=6)
        table.add_column("Summary", max_width=55)
        table.add_column("Latency", justify="right", max_width=8)

        for item in SAMPLE_CLAIMS:
            console.print(f"[dim]Verifying: {item['claim'][:70]}...[/dim]")
            try:
                data = await verify_claim(client, item["claim"], fast=False)
                result = data["result"]
                level = result["credibility_level"]
                color = LEVEL_COLORS.get(level, "white")
                icon = LEVEL_ICONS.get(level, "?")
                table.add_row(
                    item["label"],
                    f"[{color}]{icon} {level}[/{color}]",
                    f"{result['confidence_score']:.0%}",
                    result["summary"][:200],
                    f"{result.get('latency_ms', 0):.0f}ms",
                )
            except Exception as e:
                table.add_row(item["label"], "[red]ERROR[/red]", "-", str(e), "-")

        console.print()
        console.print(table)
        console.print("\n[dim]Demo complete. Full API docs at http://localhost:8000/docs[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
