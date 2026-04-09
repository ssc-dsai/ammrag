from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import psycopg2
import json
import os


class PostgreSQLSchemaInspectorInput(BaseModel):
    """Input schema for PostgreSQL Schema Inspector Tool."""
    schema_name: str = Field(default="public", description="Name of the PostgreSQL schema to inspect")
    table_filter: Optional[str] = Field(default=None, description="Optional table name prefix to filter results")


class PostgreSQLSchemaInspectorTool(BaseTool):
    """Tool for inspecting PostgreSQL database schema."""

    name: str = "PostgreSQL Schema Inspector"
    description: str = (
        "Connects to PostgreSQL and retrieves schema information including table names, "
        "column names, data types, and row counts. Returns comprehensive schema documentation "
        "to understand the available data structure for query generation."
    )
    args_schema: Type[BaseModel] = PostgreSQLSchemaInspectorInput

    def _run(self, schema_name: str = "public", table_filter: Optional[str] = None) -> str:
        """Inspect PostgreSQL schema and return structured information."""
        conn = None
        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            dbname = os.getenv("POSTGRES_DB", "appdb")
            user = os.getenv("POSTGRES_USER", "appuser")
            password = os.getenv("POSTGRES_PASSWORD", "supersecret")

            conn = psycopg2.connect(
                host=host, port=port, dbname=dbname, user=user, password=password
            )
            cur = conn.cursor()

            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                (schema_name,)
            )
            tables = [row[0] for row in cur.fetchall()]

            if table_filter:
                tables = [t for t in tables if t.startswith(table_filter)]

            if not tables:
                return f"No tables found in schema '{schema_name}'"

            result = {
                "schema": schema_name,
                "table_count": len(tables),
                "tables": {}
            }

            for table_name in tables:
                cur.execute(
                    """
                    SELECT column_name, data_type, character_maximum_length, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (schema_name, table_name)
                )
                columns = []
                for col_name, data_type, max_len, nullable in cur.fetchall():
                    col_info = {
                        "name": col_name,
                        "type": data_type,
                        "nullable": nullable == 'YES',
                    }
                    if max_len:
                        col_info["max_length"] = max_len
                    columns.append(col_info)

                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}";')
                    row_count = cur.fetchone()[0]
                except Exception:
                    row_count = "unknown"

                result["tables"][table_name] = {
                    "row_count": row_count,
                    "column_count": len(columns),
                    "columns": columns
                }

            cur.close()
            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error inspecting schema: {str(e)}"
        finally:
            if conn:
                conn.close()
