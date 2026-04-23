"""
Shared formatting for CrewAI query responses.
"""


def format_crew_response(data: dict) -> str:
    """Return the pre-formatted markdown answer produced by FormatCrew."""
    return data.get("answer") or "No answer generated."
