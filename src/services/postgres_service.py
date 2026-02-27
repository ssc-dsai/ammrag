"""
PostgreSQL metadata search service for FastAPI

Handles structured metadata queries against PostgreSQL tables.
"""

import csv
import io
import logging
import re
import uuid as uuid_lib
from datetime import datetime
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2 import sql
from fastapi import HTTPException

from src.core import settings


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
        """
        Verify PostgreSQL connectivity and ensure required tables exist.
        Tables: catalog (id, name, path), files (id, catalog_id, uuid, uri, type, last_modified_datetime, processed)
        """
        try:
            conn = self._get_connection()
        except Exception as e:
            logger.error("PostgreSQL connection FAILED: %s", e)
            return

        logger.info("PostgreSQL connection OK (%s:%s/%s)",
                     settings.postgres_host, settings.postgres_port, settings.postgres_db)

        try:
            with conn.cursor() as cur:
                # Create catalog table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS catalogs (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE NOT NULL,
                        path TEXT NOT NULL
                    )
                """)

                # Create collections table with foreign key to catalog
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS collections (
                        id SERIAL PRIMARY KEY,
                        catalog_id INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
                        name TEXT NOT NULL
                    )
                """)

                # Create files table with foreign key to catalog
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        id SERIAL PRIMARY KEY,
                        catalog_id INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
                        uuid UUID NOT NULL DEFAULT gen_random_uuid(),
                        uri TEXT NOT NULL,
                        type VARCHAR(100) NOT NULL,
                        last_modified_datetime TIMESTAMP NOT NULL,
                        processed BOOLEAN NOT NULL DEFAULT FALSE,
                        UNIQUE(catalog_id, uri)
                    )
                """)

                # Create structured table with foreign key to files
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS structured (
                        id SERIAL PRIMARY KEY,
                        file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
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

                conn.commit()
                logger.info("Database tables verified/created")
        finally:
            conn.close()

    def create_collection(self, catalog_id: int, name: str) -> int:
        """Create a new collection record for a catalog.
        Returns {"id": collection_id, "name": collection_name}.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO collections (catalog_id, name)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (catalog_id, name),
                )
                collection_id = cur.fetchone()[0]
                conn.commit()
                return collection_id
        finally:
            conn.close()

    def add_catalog(self, name: str, path: str) -> int:
        """
        Add a catalog to the database, or return the existing id if it already exists.

        Args:
            name: The name of the catalog
            path: The path of the catalog

        Returns:
            The catalog id
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO catalogs (name, path)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE SET path = EXCLUDED.path
                    RETURNING id
                    """,
                    (name, path)
                )
                row = cur.fetchone()
                conn.commit()
                assert row is not None
                return row[0]
        finally:
            conn.close()


    def get_catalogs(self, id: Optional[int] = None, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return catalogs, optionally filtered by id and/or name."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                query = "SELECT id, name, path FROM catalogs"
                conditions: list[str] = []
                params: list = []
                if id is not None:
                    conditions.append("id = %s")
                    params.append(id)
                if name is not None:
                    conditions.append("name = %s")
                    params.append(name)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY id"
                cur.execute(query, params)
                return [
                    {"id": row[0], "name": row[1], "path": row[2]}
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def get_collections(
        self, id: Optional[int] = None, catalog_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return collections, optionally filtered by id and/or catalog_id."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                query = "SELECT id, catalog_id, name FROM collections"
                conditions: list[str] = []
                params: list = []
                if id is not None:
                    conditions.append("id = %s")
                    params.append(id)
                if catalog_id is not None:
                    conditions.append("catalog_id = %s")
                    params.append(catalog_id)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY id"
                cur.execute(query, params)
                return [
                    {"id": row[0], "catalog_id": row[1], "name": row[2]}
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def get_file(self, catalog_id: Optional[int], uri: str) -> Optional[Dict[str, Any]]:
        """Return a file row matching (catalog_id, uri), or None."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, last_modified_datetime FROM files WHERE catalog_id = %s AND uri = %s",
                    (catalog_id, uri),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {"id": row[0], "last_modified": row[1]}
        finally:
            conn.close()

    def update_file_timestamp(self, file_id: int, last_modified: datetime) -> None:
        """Update the last_modified_datetime for an existing file."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE files SET last_modified_datetime = %s WHERE id = %s",
                    (last_modified, file_id),
                )
                conn.commit()
        finally:
            conn.close()

    def add_file(
        self,
        catalog_id: Optional[int],
        uri: str,
        file_type: str,
        last_modified: datetime,
        processed: bool = False,
    ) -> Optional[int]:
        """
        Add a file to the database.

        Args:
            catalog_id: The id of the catalog (None for standalone imports)
            uri: The URI path of the file
            file_type: The type/mime of the file
            last_modified: The last modified timestamp
            processed: Whether the file has been processed

        Returns:
            The file id if created, None if the file already exists
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO files (catalog_id, uri, type, last_modified_datetime, processed)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (catalog_id, uri) DO NOTHING
                    RETURNING id
                    """,
                    (catalog_id, uri, file_type, last_modified, processed),
                )
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else None
        finally:
            conn.close()


    async def add_structured(self, file_id: int, csv_data: str, table_name: str | None = None) -> str:
        """
        Create a PostgreSQL table from CSV data and register it in the structured table.

        Args:
            file_id: The id of the file in the files table
            csv_data: Raw CSV string (first row is headers)
            table_name: Optional explicit table name; auto-generated if not provided

        Returns:
            The table name
        """
        if table_name is None:
            uid = uuid_lib.uuid4().hex[:12]
            table_name = f"file_{file_id}_{uid}"

        reader = csv.reader(io.StringIO(csv_data))
        headers = [re.sub(r"\W+", "_", h.strip().lower()) for h in next(reader)]
        rows = list(reader)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Build CREATE TABLE with all TEXT columns
                cols = sql.SQL(", ").join(
                    sql.SQL("{} TEXT").format(sql.Identifier(h)) for h in headers
                )
                cur.execute(sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(table_name), cols
                ))

                # Insert rows
                if rows:
                    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(headers))
                    insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(table_name),
                        sql.SQL(", ").join(sql.Identifier(h) for h in headers),
                        placeholders,
                    )
                    cur.executemany(insert, rows)

                # Register in structured table
                cur.execute(
                    "INSERT INTO structured (file_id, table_name) VALUES (%s, %s)",
                    (file_id, table_name),
                )

                conn.commit()
                return table_name
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


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

    def get_structured_tables(self, file_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Get structured table registrations, optionally filtered by file IDs.

        Args:
            file_ids: Optional list of file IDs to filter by

        Returns:
            List of dicts with keys: id, file_id, table_name
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if file_ids:
                    placeholders = ", ".join(["%s"] * len(file_ids))
                    cur.execute(
                        f"SELECT id, file_id, table_name FROM structured WHERE file_id IN ({placeholders}) ORDER BY id",
                        file_ids,
                    )
                else:
                    cur.execute("SELECT id, file_id, table_name FROM structured ORDER BY id")
                return [
                    {"id": row[0], "file_id": row[1], "table_name": row[2]}
                    for row in cur.fetchall()
                ]
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
    