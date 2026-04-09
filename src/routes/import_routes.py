"""
Import routes for file import operations
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

from src.schemas.file_schemas import ImportResponse, ImportJobResponse
from src.core.config import settings
from src.services.file_service import file_service

router = APIRouter(
    prefix="/import",
    tags=["import"],
    responses={
        400: {"description": "Bad request"},
        409: {"description": "Import already running"},
        422: {"description": "Validation error"},
    },
)


# ---------------------------------------------------------------------------
# In-process job tracker
# ---------------------------------------------------------------------------

@dataclass
class _ImportJob:
    status: str                          # running | completed | failed
    projects: List[str]
    started_at: datetime
    files_total: int = 0
    completed_at: Optional[datetime] = None
    files: List[ImportResponse] = field(default_factory=list)
    error: Optional[str] = None

    def to_response(self) -> ImportJobResponse:
        return ImportJobResponse(
            status=self.status,
            projects=self.projects,
            started_at=self.started_at,
            completed_at=self.completed_at,
            files_total=self.files_total,
            files_processed=len(self.files),
            files=self.files if self.status != "running" else [],
            error=self.error,
        )


_current_job: Optional[_ImportJob] = None


async def _run_import(job: _ImportJob, project_names: List[str]) -> None:
    """Background coroutine that performs the import and updates job state."""
    try:
        # Collect all URIs first so we can report files_total upfront
        from urllib.parse import urlparse
        all_projects = []
        for name in project_names:
            all_projects.extend(p for p in settings.projects if p["name"] == name)

        all_uris: List[str] = []
        for project in all_projects:
            parsed = urlparse(project["path"])
            if parsed.scheme in ('http', 'https'):
                uris = file_service._crawl_web_directory(project["path"])
            else:
                local_path = parsed.path if parsed.scheme == 'file' else project["path"]
                uris = file_service._walk_local_directory(local_path)
            all_uris.extend(uris)

        job.files_total = len(all_uris)

        def _on_file(result: ImportResponse) -> None:
            job.files.append(result)

        failed_projects = []
        for project in all_projects:
            try:
                await file_service.import_batch_files(
                    project["path"],
                    project_name=project["name"],
                    on_file_imported=_on_file,
                )
            except Exception as exc:
                logger.exception("Import failed for project '%s': %s", project["name"], exc)
                failed_projects.append(project["name"])

        if failed_projects:
            job.status = "completed_with_errors"
            job.error = f"Failed projects: {', '.join(failed_projects)}"
        else:
            job.status = "completed"
    except Exception as exc:
        logger.exception("Import job failed: %s", exc)
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.completed_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/project",
    response_model=ImportJobResponse,
    summary="Start import from project(s)",
    description=(
        "Launch a background import of all files from a specific project or all projects. "
        "Returns immediately with job status. Poll /import/status for progress."
    ),
    status_code=202,
)
async def import_project(
    name: Optional[str] = Query(None, description="Project name to import (omit for all)"),
) -> ImportJobResponse:
    global _current_job

    if _current_job is not None and _current_job.status == "running":
        raise HTTPException(status_code=409, detail="An import is already running")

    projects = settings.projects
    if name is not None:
        projects = [p for p in projects if p["name"] == name]

    if not projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project_names = [p["name"] for p in projects]
    _current_job = _ImportJob(
        status="running",
        projects=project_names,
        started_at=datetime.now(timezone.utc),
    )

    asyncio.create_task(_run_import(_current_job, project_names))

    return _current_job.to_response()


@router.get(
    "/status",
    response_model=ImportJobResponse,
    summary="Import job status",
    description="Show the status of the currently running import, or the results of the last import.",
)
async def import_status() -> ImportJobResponse:
    if _current_job is None:
        return ImportJobResponse(
            status="idle",
            projects=[],
            started_at=None,
            completed_at=None,
            files_total=0,
            files_processed=0,
            error=None,
        )
    return _current_job.to_response()
