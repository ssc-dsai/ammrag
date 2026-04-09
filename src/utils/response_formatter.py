"""
Shared formatting for CrewAI query responses.
"""


def format_crew_response(data: dict) -> str:
    """Format a /crews/query response dict as human-readable markdown."""
    aspects = data.get("aspects") or []
    if not aspects:
        return "No answer generated."

    lines = []
    for i, aspect in enumerate(aspects, 1):
        lines.append(f"### {i}. {aspect.get('discussion', '')}")
        quote = aspect.get("quote", "")
        if quote:
            lines.append(f"> {quote}")
        source = aspect.get("source", "")
        if source and source != "not found":
            lines.append(f"*Source: {source}*")
        lines.append("")

    sources = data.get("sources") or []
    if sources:
        lines.append("---\n**Sources**")
        lines.extend(f"- {s}" for s in sources)

    return "\n".join(lines)
