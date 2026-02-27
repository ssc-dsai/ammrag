import logging
from pathlib import Path
from typing import Optional

import yaml
from watchdog.events import (
    FileSystemEventHandler,
    DirCreatedEvent,
    FileCreatedEvent,
    DirDeletedEvent,
    FileDeletedEvent,
    DirModifiedEvent,
    FileModifiedEvent,
    DirMovedEvent,
    FileMovedEvent,
)

from email_assistant import models
from email_assistant.crew import KnowledgeOrganizingCrew

logger = logging.getLogger(__name__)


class AgenticFileSystemToQdrantHandler(FileSystemEventHandler):
    """
    An event handler for the changes done in the file system. It handles the synchronization between
    the filesystem and the Qdrant knowledge base used by the crew.
    """

    def __init__(
        self,
        embedder_config: dict,
        qdrant_location: str,
        qdrant_api_key: Optional[str] = None,
        min_content_length: int = 10,
    ):
        crew = KnowledgeOrganizingCrew(embedder_config, qdrant_location, qdrant_api_key)
        self.crew = crew.crew()
        self.knowledge_base = crew.knowledge_base()
        self.min_content_length = min_content_length

    def initialize(self, init_path: Path):
        """
        Initialize the Qdrant collection with existing files.
        """
        for file_path in Path(init_path).rglob("*.jsonl"):
            file_path_str = str(file_path)
            points_count = self.knowledge_base.count({"src_path": file_path_str})
            if points_count > 0:
                logger.info("File already exists in the knowledge base: %s", file_path)
                continue

            self.on_created(
                FileCreatedEvent(file_path_str, file_path_str, is_synthetic=True)
            )

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """
        Load a new file content into Qdrant knowledge base. Ignore new directories.
        :param event:
        :return:
        """
        if isinstance(event, DirCreatedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".jsonl"):
            return

        # Log the event
        logger.info("New file created: %s", event.src_path)

        # Load the file content
        file_content = Path(event.src_path).read_text().strip()

        # Only process the file if the content is longer than the minimum length
        if len(file_content) < self.min_content_length:
            logger.info(
                "The file content is shorter than the minimum length of %i: %s",
                self.min_content_length,
                event.src_path,
            )
            return

        # Load the frontmatter from the Markdown file
        try:
            frontmatter = next(yaml.safe_load_all(file_content))
            if not isinstance(frontmatter, dict):
                frontmatter = {}
        except StopIteration:
            frontmatter = {}

        # Run the knowledge organizing crew to store the file content in the knowledge base
        response = self.crew.kickoff(
            inputs={"src_path": event.src_path, "document": file_content}
        )
        if not isinstance(response.pydantic, models.ContextualizedChunks):
            logger.info("Did not receive any contextualized chunks: %s", response)
            return

        # Store the response in the Qdrant knowledge base
        document_chunks: models.ContextualizedChunks = response.pydantic  # noqa
        for chunk in document_chunks.chunks:
            formatted_input_data = f"{chunk.content}\n\n{chunk.context}"
            metadata = {
                "src_path": event.src_path,
                "chunk_context": chunk.context,
                "chunk_content": chunk.content,
                **frontmatter,
            }
            self.knowledge_base.save(formatted_input_data, metadata)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        """
        When the file is removed, all its content has to be removed from Qdrant as well.
        If a directory is removed, there is a separate event triggered for all the files inside, so
        we ignore directories.
        :param event:
        :return:
        """
        if isinstance(event, DirDeletedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Log the event
        logger.info("File deleted: %s", event.src_path)

        # Remove all the entries related to the file from the knowledge base
        self.knowledge_base.delete({"src_path": event.src_path})

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """
        When the file is modified, remove all the existing content related to this file in Qdrant,
        and then load the new content. Ignore directories, as modifications to directories themselves
        do not mean anything in terms of the content.
        :param event:
        :return:
        """
        if isinstance(event, DirModifiedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Log the event
        logger.info("File modified: %s", event.src_path)

        # Remove the existing content
        self.on_deleted(
            FileDeletedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content
        self.on_created(
            FileCreatedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """
        Update the file path in Qdrant knowledge base. Ignore directories.
        :param event:
        :return:
        """
        if isinstance(event, DirMovedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Log the event
        logger.info("File moved: %s -> %s", event.src_path, event.dest_path)

        # Remove the existing content from the old path
        self.on_deleted(
            FileDeletedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content from the new location
        self.on_created(
            FileCreatedEvent(event.dest_path, event.dest_path, is_synthetic=True)
        )
