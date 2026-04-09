"""
PostgreSQL metadata search service for FastAPI

Handles structured metadata queries against PostgreSQL tables.
"""

import logging
import re
import uuid as uuid_lib
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2 import sql
from fastapi import HTTPException

from src.core import settings
from src.models.ollama_models import ParsedTable


logger = logging.getLogger(__name__)


class PostgresService:
    """Service for PostgreSQL metadata operations"""

    def __init__(self) -> None:
        self._ensure_tables()
        

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )

    async def list_tables(self) -> List[str]:
        """
        List all user tables in the public schema.

        Returns:
            List of table names
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' ORDER BY table_name"
                    )
                    return [row[0] for row in cur.fetchall()]
            finally:
                conn.close()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"PostgreSQL error: {str(e)}"
            )


    def _ensure_tables(self) -> None:
        """Verify PostgreSQL connectivity and ensure the structured table exists."""
        try:
            conn = self._get_connection()
        except Exception as e:
            logger.error("PostgreSQL connection FAILED: %s", e)
            return

        logger.info("PostgreSQL connection OK (%s:%s/%s)",
                     settings.postgres_host, settings.postgres_port, settings.postgres_db)

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS structured (
                        id SERIAL PRIMARY KEY,
                        file_uuid TEXT NOT NULL,
                        table_name TEXT NOT NULL
                    )
                """)

                # Trigger to drop the dynamic table when its structured row is deleted
                cur.execute("""
                    CREATE OR REPLACE FUNCTION drop_structured_table()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        EXECUTE format('DROP TABLE IF EXISTS %I', OLD.table_name);
                        RETURN OLD;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                cur.execute("""
                    DROP TRIGGER IF EXISTS trg_drop_structured_table ON structured;
                    CREATE TRIGGER trg_drop_structured_table
                        BEFORE DELETE ON structured
                        FOR EACH ROW
                        EXECUTE FUNCTION drop_structured_table();
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS filelist (
                        id SERIAL PRIMARY KEY,
                        root_dir TEXT NOT NULL,
                        collection_name TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        UNIQUE (root_dir, collection_name, uri)
                    )
                """)

                conn.commit()
                logger.info("Database tables verified/created")
        finally:
            conn.close()

    def _insert_parsed_table(self, parsed: ParsedTable, file_id: str) -> str:
        """Insert a single ParsedTable into Postgres. Returns the generated UUID table name."""
        table_id = uuid_lib.uuid4().hex
        headers = [re.sub(r"\W+", "_", h.strip().lower()) for h in parsed.headers]
        rows = parsed.rows

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cols = sql.SQL(", ").join(
                    sql.SQL("{} TEXT").format(sql.Identifier(h)) for h in headers
                )
                cur.execute(sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(table_id), cols
                ))

                if rows:
                    n = len(headers)
                    padded = [tuple((row + [""] * n)[:n]) for row in rows]
                    placeholders = sql.SQL(", ").join([sql.Placeholder()] * n)
                    insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(table_id),
                        sql.SQL(", ").join(sql.Identifier(h) for h in headers),
                        placeholders,
                    )
                    cur.executemany(insert.as_string(cur), padded)

                cur.execute(
                    "INSERT INTO structured (file_uuid, table_name) VALUES (%s, %s)",
                    (file_id, table_id),
                )
                conn.commit()
                return table_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def add_structured(self, file_id: str, csv_data: str) -> list[str]:
        """
        Parse a raw CSV with Ollama and store each resulting table in Postgres
        with a UUID table name.

        Returns:
            List of UUID table names created (one per parsed sub-table),
            or an empty list if tables already exist for this file.
        """
        from src.services.ollama_service import ollama_service  # local import avoids circular

        # Skip if structured tables already exist for this file
        # conn = self._get_connection()
        # try:
        #     with conn.cursor() as cur:
        #         cur.execute("SELECT 1 FROM structured WHERE file_uuid = %s", (file_id,))
        #         if cur.fetchone():
        #             logger.info("Structured tables already exist for '%s' — skipping", file_id)
        #             return []
        # finally:
        #     conn.close()

        parsed_tables = await ollama_service.parse_csv_tables(csv_data)

        table_ids: list[str] = []
        for parsed in parsed_tables:
            table_id = self._insert_parsed_table(parsed, file_id)
            table_ids.append(table_id)
            logger.info("Stored structured table '%s' (title: %s)", table_id, parsed.title)

        return table_ids


    async def search_metadata(
        self,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Search structured metadata tables.

        If table_name is provided, queries that specific table.
        If query looks like a SQL SELECT statement, executes it directly.
        Otherwise performs a text search across columns.

        Args:
            query: Text search term or SQL SELECT statement
            table_name: Specific table to query
            filters: Key-value filters for WHERE clause
            limit: Maximum rows to return

        Returns:
            Dict with keys: columns, rows, row_count
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if not table_name:
                    # Return list of structured tables
                    cur.execute(
                        "SELECT table_name FROM structured ORDER BY id"
                    )
                    tables = [row[0] for row in cur.fetchall()]
                    return {"columns": ["table_name"], "rows": [[t] for t in tables], "row_count": len(tables)}

                # Validate table exists
                cur.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
                    (table_name,),
                )
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

                # Build query
                if query and query.strip().upper().startswith("SELECT"):
                    # Execute raw SELECT query (read-only)
                    cur.execute(query)
                else:
                    # Get columns and do text search
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = %s",
                        (table_name,),
                    )
                    columns = [row[0] for row in cur.fetchall()]

                    where_parts: list[str] = []
                    params: list = []

                    if query:
                        text_conditions = []
                        for col in columns:
                            text_conditions.append(f"{col}::TEXT ILIKE %s")
                            params.append(f"%{query}%")
                        if text_conditions:
                            where_parts.append(f"({' OR '.join(text_conditions)})")

                    if filters:
                        for k, v in filters.items():
                            where_parts.append(f"{k} = %s")
                            params.append(v)

                    stmt = f"SELECT * FROM {table_name}"
                    if where_parts:
                        stmt += " WHERE " + " AND ".join(where_parts)
                    stmt += f" LIMIT {limit}"
                    cur.execute(stmt, params)

                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                return {
                    "columns": columns,
                    "rows": [list(row) for row in rows],
                    "row_count": len(rows),
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PostgreSQL error: {str(e)}")
        finally:
            conn.close()

    async def execute_select(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a read-only SELECT query against PostgreSQL.

        Args:
            sql_query: A SQL SELECT statement

        Returns:
            Dict with keys: columns, rows, row_count

        Raises:
            HTTPException if the query is not a SELECT or on error
        """
        stripped = sql_query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(stripped)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                return {
                    "columns": columns,
                    "rows": [list(row) for row in rows],
                    "row_count": len(rows),
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PostgreSQL error: {str(e)}")
        finally:
            conn.close()

    def get_structured_tables(self, file_uuids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get structured table registrations, optionally filtered by file UUIDs.

        Args:
            file_uuids: Optional list of file UUIDs to filter by

        Returns:
            List of dicts with keys: id, file_uuid, table_name
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if file_uuids:
                    placeholders = ", ".join(["%s"] * len(file_uuids))
                    cur.execute(
                        f"SELECT id, file_uuid, table_name FROM structured WHERE file_uuid IN ({placeholders}) ORDER BY id",
                        file_uuids,
                    )
                else:
                    cur.execute("SELECT id, file_uuid, table_name FROM structured ORDER BY id")
                return [
                    {"id": row[0], "file_uuid": row[1], "table_name": row[2]}
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def delete_structured_by_file_id(self, file_id: str) -> int:
        """Delete all structured table registrations for a file, cascading to drop the dynamic tables.

        The DB trigger drop_structured_table fires BEFORE DELETE on the structured
        registry and drops each dynamic table automatically.

        Returns:
            Number of rows deleted from the structured registry.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM structured WHERE file_uuid = %s", (file_id,))
                count = cur.rowcount
                conn.commit()
                if count:
                    logger.info("Deleted %d structured table(s) for file '%s'", count, file_id)
                return count
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"PostgreSQL error: {str(e)}")
        finally:
            conn.close()

    def get_filelist(self, root_dir: str, collection_name: str) -> List[str]:
        """Return the stored URI list for a root_dir/collection pair, or [] if none exists."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT uri FROM filelist WHERE root_dir = %s AND collection_name = %s ORDER BY uri",
                    (root_dir, collection_name),
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def set_filelist(self, root_dir: str, collection_name: str, uris: List[str]) -> None:
        """Replace the stored URI list for a root_dir/collection pair."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM filelist WHERE root_dir = %s AND collection_name = %s",
                    (root_dir, collection_name),
                )
                if uris:
                    cur.executemany(
                        "INSERT INTO filelist (root_dir, collection_name, uri) VALUES (%s, %s, %s) "
                        "ON CONFLICT DO NOTHING",
                        [(root_dir, collection_name, uri) for uri in uris],
                    )
                conn.commit()
                logger.info("Updated filelist for '%s' (%s): %d file(s)", root_dir, collection_name, len(uris))
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"PostgreSQL error: {str(e)}")
        finally:
            conn.close()

    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get column names for a table.

        Args:
            table_name: The table name

        Returns:
            List of column names
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (table_name,),
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()


# Global service instance
postgres_service = PostgresService()

if __name__ == "__main__":
    postgres_service._ensure_tables()
    