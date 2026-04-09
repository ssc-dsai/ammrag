from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import psycopg2
import psycopg2.extras
import json
import os
from datetime import datetime, date
from decimal import Decimal


class PostgreSQLQueryExecutorInput(BaseModel):
    """Input schema for PostgreSQL Query Executor Tool."""
    sql_query: str = Field(..., description="The SQL SELECT query to execute against PostgreSQL")
    max_rows: int = Field(default=100, description="Maximum number of rows to return (default 100)")
    schema_name: str = Field(default="public", description="Default schema for the search_path")


class PostgreSQLQueryExecutorTool(BaseTool):
    """Tool for executing SQL queries against PostgreSQL and returning results."""

    name: str = "PostgreSQL Query Executor"
    description: str = (
        "Executes SQL SELECT queries against a PostgreSQL database and returns the results "
        "in a structured format. Handles data type conversion and limits result sets for readability. "
        "Use this tool to run SQL queries that answer the user's natural language question."
    )
    args_schema: Type[BaseModel] = PostgreSQLQueryExecutorInput

    def _serialize_value(self, value):
        """Convert PostgreSQL result values to JSON-serializable types."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def _run(self, sql_query: str, max_rows: int = 100, schema_name: str = "public") -> str:
        """Execute SQL query and return results."""
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
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(f"SET search_path TO {schema_name}, public;")

            query_stripped = sql_query.strip().rstrip(';')
            query_upper = query_stripped.upper()
            if query_upper.startswith('SELECT') and 'LIMIT' not in query_upper:
                limited_query = f"{query_stripped} LIMIT {max_rows};"
            else:
                limited_query = sql_query

            cur.execute(limited_query)
            rows = cur.fetchall()

            if not rows:
                return json.dumps({
                    "success": True,
                    "query": sql_query,
                    "row_count": 0,
                    "columns": [],
                    "data": [],
                    "message": "Query executed successfully but returned no rows."
                })

            column_names = list(rows[0].keys())
            data = [
                {k: self._serialize_value(v) for k, v in row.items()}
                for row in rows
            ]

            result = {
                "success": True,
                "query": sql_query,
                "row_count": len(data),
                "columns": column_names,
                "data": data
            }

            if len(data) == max_rows:
                result["note"] = f"Results limited to {max_rows} rows. Use a more specific query for complete data."

            cur.close()
            return json.dumps(result, indent=2, ensure_ascii=False)

        except psycopg2.Error as e:
            return json.dumps({
                "success": False,
                "query": sql_query,
                "error": str(e),
                "error_type": type(e).__name__
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "query": sql_query,
                "error": f"Unexpected error: {str(e)}"
            })
        finally:
            if conn:
                conn.close()
