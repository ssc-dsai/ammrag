"""
Send a CSV file to ollama and get back structured markdown tables.
"""

import csv
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel

load_dotenv()


class MarkdownTable(BaseModel):
    title: str
    markdown: str


class TableList(BaseModel):
    tables: list[MarkdownTable]


PROMPT = """\
You are given several CSV blocks, each separated by a blank line and preceded \
by a "--- Section N ---" marker. Each block is already a clean, self-contained \
table with no blank columns.

Your task:
1. Convert each CSV block into a valid Markdown table.
2. The first row of each block is the header row.
3. Collapse any remaining multiline cell values into a single line \
(replace newlines with a space).
4. Give each table a short descriptive title.
5. Do not invent rows or duplicate data.

Return ONLY the structured JSON.

CSV blocks:
{csv_content}
"""


def collapse_newlines(value: str) -> str:
    return " ".join(value.split())


def parse_csv(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text))
    return [[collapse_newlines(cell) for cell in row] for row in reader]


def split_into_sections(rows: list[list[str]]) -> list[list[list[str]]]:
    """Split rows into sections on blank rows."""
    sections: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            if current:
                sections.append(current)
                current = []
        else:
            current.append(row)
    if current:
        sections.append(current)
    return sections


def drop_empty_cols(rows: list[list[str]]) -> list[list[str]]:
    """Remove columns that are entirely empty."""
    if not rows:
        return rows
    num_cols = max(len(r) for r in rows)
    padded = [r + [""] * (num_cols - len(r)) for r in rows]
    keep = [c for c in range(num_cols) if any(padded[r][c].strip() for r in range(len(padded)))]
    return [[row[c] for c in keep] for row in padded]


def normalize_grouped_header(rows: list[list[str]]) -> list[list[str]]:
    """
    When the header row's col 0 is a group/section label and col 1 is a
    field-name label (e.g. "Room #"), replace them with generic names
    "Section" and "Field" so the LLM understands the column roles clearly.
    The data values in cols 2+ keep their original header names.
    """
    if not rows:
        return rows
    header = list(rows[0])
    # Only apply when there are ≥3 columns and col 1 looks like a row-label,
    # i.e. it's a short label and NOT a room/entity identifier.
    if len(header) >= 3:
        header[0] = "Section"
        header[1] = "Field"
        return [header] + rows[1:]
    return rows


def forward_fill_col0(rows: list[list[str]]) -> list[list[str]]:
    """
    Forward-fill empty cells in column 0.
    Spreadsheets often put a section label only on the first row of a group;
    filling it down makes every row self-contained for the LLM.
    Skip the header row (index 0).
    """
    if len(rows) < 2:
        return rows
    result = [rows[0]]
    # Seed from the header row's col-0 so rows immediately below inherit it
    last = rows[0][0].strip() if rows[0] else ""
    for row in rows[1:]:
        val = row[0].strip() if row else ""
        if val:
            last = val
        else:
            row = [last] + row[1:]
        result.append(row)
    return result


def rows_to_csv_text(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return buf.getvalue().strip()


def load_and_clean_csv(csv_path: str) -> str:
    content = Path(csv_path).read_text(encoding="utf-8")
    rows = parse_csv(content)

    # Drop trailing all-empty rows
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()

    sections = split_into_sections(rows)

    parts: list[str] = []
    for i, section in enumerate(sections, 1):
        cleaned = drop_empty_cols(section)
        # Forward-fill the grouping column if section has ≥3 columns
        # (heuristic: a 2-col section is a simple key-value table, not grouped)
        if cleaned and max(len(r) for r in cleaned) >= 3:
            cleaned = forward_fill_col0(cleaned)
            cleaned = normalize_grouped_header(cleaned)
        parts.append(f"--- Section {i} ---\n{rows_to_csv_text(cleaned)}")

    return "\n\n".join(parts)


def csv_to_markdown(csv_path: str, model: str | None = None) -> list[MarkdownTable]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "gemma4:12B")
    client = Client(host=host)

    cleaned_csv = load_and_clean_csv(csv_path)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(csv_content=cleaned_csv),
            }
        ],
        format=TableList.model_json_schema(),
    )

    result = TableList.model_validate_json(response.message.content)
    return result.tables


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    model = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Processing: {csv_path}  (model: {model or os.getenv('OLLAMA_MODEL')})\n")
    tables = csv_to_markdown(csv_path, model)

    for i, table in enumerate(tables, 1):
        print(f"### Table {i}: {table.title}\n")
        print(table.markdown)
        print()


if __name__ == "__main__":
    main()
