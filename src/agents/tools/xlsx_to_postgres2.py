import re
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Tuple
import os
from ollama import chat, Client
from pydantic import BaseModel
from typing import Optional 
import requests
import logging

logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
CACHE_PATH = os.getenv(
    "CACHE_PATH",
    "/workspaces/canchat-v2-nss/nss/crewai/data/cache",
)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://appuser:supersecret@192.168.68.60:5432/appdb",
)

# if_exists options: "replace", "append", "fail"
IF_EXISTS = "replace"

# Allowed: append only (no destructive behavior)
# IF_EXISTS = "append"
SCHEMA = os.getenv("DB_SCHEMA", "public")
# ----------------------------------------

# ---------------- Models ----------------

class OllamaCSVSubtableParseRequest(BaseModel):
  role: Optional[str] = 'user'
  question: str
  csv_content: str
  
class OllamaCSVSubtableParseResponse(BaseModel):
  answer: bool
  reasoning: str


# --------------------------------------------
def normalize_name(name: str) -> str:
    """
    Normalize table/column names:
    - lowercase
    - replace spaces & invalid chars with _
    - strip leading digits
    """
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"^(\d+)", r"_\1", name)
    return name
def get_opinion(request: OllamaCSVSubtableParseRequest):
    # Retrieve server info and model from environment variables
    # model_name = os.getenv("OLLAMA_MODEL")
    model_name = "llama3.2:latest"

    response = chat(
        messages=[
            {
            'role': request.role,
            'content': request.question + '\n\n' +request.csv_content,
            }
        ],
        model=model_name,
        format=OllamaCSVSubtableParseResponse.model_json_schema()
        )
    if response.message.content is None:
        raise ValueError("Ollama response content is None")
    return OllamaCSVSubtableParseResponse.model_validate_json(response.message.content)




def list_xlsx_files_in_cache() -> List[Tuple[str, List[Path]]]:
    """
    Returns:
        [(location_name, [xlsx_path, ...]), ...]
    """
    base = Path(CACHE_PATH)
    if not base.is_dir():
        raise RuntimeError(f"CACHE_PATH does not exist: {CACHE_PATH}")

    results: List[Tuple[str, List[Path]]] = []

    for location_dir in base.iterdir():
        surveys_dir = location_dir / "surveys"
        if not surveys_dir.is_dir():
            continue

        xlsx_files = list(surveys_dir.glob("*.xlsx"))
        if xlsx_files:
            results.append((location_dir.name, xlsx_files))

    return results


def ensure_location_index(engine, location_name):
    """
    Ensure the location_name is present in the location_index table.
    Returns the unique index number for the location.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        # Create table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS location_index (
                id SERIAL PRIMARY KEY,
                location_name TEXT UNIQUE NOT NULL
            )
        """))
        # Try to insert, ignore if exists
        conn.execute(text("""
            INSERT INTO location_index (location_name)
            VALUES (:location_name)
            ON CONFLICT (location_name) DO NOTHING
        """), {"location_name": location_name})
        # Fetch the id
        result = conn.execute(text("""
            SELECT id FROM location_index WHERE location_name = :location_name
        """), {"location_name": location_name})
        row = result.fetchone()
        return row[0] if row else None

# ---------------- DATABASE VALIDATION ----------------


def validate_required_tables(engine: Engine) -> None:
    """
    Ensure required schema objects exist.
    Fail fast instead of mutating schema.
    """
    required_tables = {"location_index"}

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = :schema
            """),
            {"schema": SCHEMA},
        )

        existing = {row[0] for row in result}
        missing = required_tables - existing

        if missing:
            raise RuntimeError(
                f"Missing required tables in schema '{SCHEMA}': {missing}. "
                "Run database migrations before starting the importer."
            )

def update_file_index(engine):
    """
    For each subdirectory in CACHE_PATH:
      - Add subdir name to location_index if not present.
      - For each file (recursively, excluding 'Zone.Identifier'), add/update file_index with:
        location_id, file_path, file_ext, last_write.
    """
    from sqlalchemy import text
    import datetime

    if not os.path.isdir(CACHE_PATH):
        print(f"Error: CACHE_PATH does not exist: {CACHE_PATH}", file=sys.stderr)
        return

    with engine.begin() as conn:
        for subdir in os.listdir(CACHE_PATH):
            subdir_path = os.path.join(CACHE_PATH, subdir)
            if not os.path.isdir(subdir_path):
                continue
            location_name = subdir
            # Ensure location_index entry
            conn.execute(text("""
                INSERT INTO location_index (location_name)
                VALUES (:location_name)
                ON CONFLICT (location_name) DO NOTHING
            """), {"location_name": location_name})
            # Get location_id
            result = conn.execute(text("""
                SELECT id FROM location_index WHERE location_name = :location_name
            """), {"location_name": location_name})
            row = result.fetchone()
            if not row:
                continue
            location_id = row[0]
            # Walk files recursively
            for root, dirs, files in os.walk(subdir_path):
                for fname in files:
                    if "Zone.Identifier" in fname:
                        continue
                    fpath = os.path.join(root, fname)
                    file_ext = os.path.splitext(fname)[1].lower()
                    last_write = datetime.datetime.fromtimestamp(os.path.getmtime(fpath)).astimezone()
                    # Insert or update file_index
                    conn.execute(text("""
                        INSERT INTO file_index (location_id, file_path, file_ext, last_write)
                        VALUES (:location_id, :file_path, :file_ext, :last_write)
                        ON CONFLICT (file_path) DO UPDATE
                        SET last_write = EXCLUDED.last_write,
                            location_id = EXCLUDED.location_id,
                            updated = TRUE
                        WHERE file_index.last_write IS DISTINCT FROM EXCLUDED.last_write
                    """), {
                        "location_id": location_id,
                        "file_path": fpath,
                        "file_ext": file_ext,
                        "last_write": last_write,
                    })

# ---------------- MAIN INGESTION ----------------

def main() -> None:

    # Load test.csv content
    with open("/workspaces/canchat-v2-nss/nss/crewai/nss_agents/src/nss_agents/tools/test.csv", "r", encoding="utf-8") as f:
        csv_content = f.read()

    req = OllamaCSVSubtableParseRequest(
        question="How many subtables are in the following CSV content?",
        csv_content=csv_content
    )
    opinion = get_opinion(req)
    print(opinion.answer)
    print(opinion.reasoning)

    engine = create_engine(DATABASE_URL, future=True)

    update_file_index(engine)

    # iterate through xlsx files in file_index where file_ext = '.xlsx' and updated = TRUE
    with engine.begin() as conn:
        result = conn.execute(text(f"""
            SELECT id, file_path
            FROM file_index
            WHERE file_ext = '.xlsx' AND updated = TRUE
        """))
        xlsx_files = result.fetchall()

    for id, file_path in xlsx_files:
        print(f"Loading Excel file: {file_path}")
        if not os.path.isfile(file_path):
            print(f"Error: Excel file does not exist: {file_path}", file=sys.stderr)
            continue

        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            
            # Skip sheets with ' Ref' or ' Checklist' in the name
            skip_keywords = [' Ref', 'Photo Checklist', 'DataValidation']
            sheet_name_str = str(sheet_name)
            if any(word in sheet_name_str for word in skip_keywords):
                print(f"Skipping sheet: {sheet_name}")
                continue

            # add the sheet to structured_data_index
            with engine.begin() as conn:
                result = conn.execute(text("""
                    INSERT INTO structured_data_index (file_id, table_title)
                    VALUES (:file_id, :table_title)
                    ON CONFLICT (file_id, table_title) DO NOTHING
                    RETURNING table_uuid
                """), {"file_id": id, "table_title": sheet_name})
                row = result.fetchone()
                if row:
                    table_uuid = row[0]
                else:
                    # If already exists, fetch the uuid
                    result = conn.execute(text("""
                        SELECT table_uuid FROM structured_data_index
                        WHERE file_id = :file_id AND table_title = :table_title
                    """), {"file_id": id, "table_title": sheet_name})
                    table_uuid = result.scalar()

            
            print(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [normalize_name(c) for c in df.columns]
            table_name = f"file_{id}_{table_uuid}"

            df.to_sql(
                table_name,
                engine,
                schema=SCHEMA,
                if_exists=IF_EXISTS,
                index=False,
                method="multi",
            )
            print(f"  → Imported into table: {SCHEMA}.{table_name}")

        # Mark file as processed
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE file_index
                SET updated = FALSE
                WHERE id = :id
            """), {"id": id})  

        print("Import completed successfully.")


if __name__ == "__main__":
    main()
