from datetime import datetime, timezone
from typing import List


def generate_report(query: str, answer: str, sources: List[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    findings = answer.split("\n")
    bullet_points = ""
    for line in findings:
        stripped = line.strip()
        if stripped:
            if stripped.startswith("-") or stripped.startswith("•"):
                bullet_points += f"{stripped}\n"
            else:
                bullet_points += f"- {stripped}\n"

    sources_section = ""
    for i, src in enumerate(sources, 1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        source_type = src.get("source_type", "unknown")
        score = src.get("relevance_score", 0.0)
        if url:
            sources_section += (
                f"{i}. [{title}]({url}) — *{source_type}* "
                f"(relevance: {score:.2f})\n"
            )
        else:
            sources_section += (
                f"{i}. {title} — *{source_type}* "
                f"(relevance: {score:.2f})\n"
            )

    report = f"""# Research Report

## Query
{query}

## Summary
{answer}

## Key Findings
{bullet_points}

## Sources
{sources_section}

## Generated on
{now}
"""
    return report
