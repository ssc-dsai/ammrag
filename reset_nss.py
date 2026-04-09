"""
Reset script: clears the 'nss' Qdrant collection and all NSS Postgres tables,
then launches the main application.

Usage:
    python reset_nss.py
"""

import subprocess
import sys

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings

COLLECTION = settings.mcp_collection_name

POSTGRES_SQL = """
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename LIKE '%'
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
"""


def reset_qdrant() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        **({"api_key": settings.qdrant_api_key} if settings.qdrant_api_key else {}),
    )
    try:
        client.delete_collection(COLLECTION)
        print(f"Qdrant: deleted collection '{COLLECTION}'")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"Qdrant: collection '{COLLECTION}' not found, nothing to delete")
        else:
            raise


def reset_postgres() -> None:
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(POSTGRES_SQL)
        conn.commit()
        print("Postgres: dropped all tables")
    finally:
        conn.close()


if __name__ == "__main__":
    print("--- Resetting NSS data ---")
    reset_qdrant()
    reset_postgres()
    print("--- Reset complete, starting application ---")
    sys.exit(subprocess.run([sys.executable, "main.py"]).returncode)
