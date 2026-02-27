"""
Upsert crew scaffold for future ingestion/indexing pipeline.

NOT connected to MCP server. Triggered by cron jobs or internal events.
This is a skeleton — tools and full task logic are not yet implemented.
"""

import logging

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

logger = logging.getLogger(__name__)


@CrewBase
class UpsertCrew:
    """
    Future crew for managing document upsert operations:
    - Vector Updater: handles embedding and upserting to Qdrant
    - Index Manager: manages PostgreSQL metadata index entries

    This crew is NOT called by the MCP server. It is intended to be
    triggered by cron jobs, internal system events, or FastAPI callbacks.
    """

    agents_config = "config/upsert/agents.yaml"
    tasks_config = "config/upsert/tasks.yaml"

    @agent
    def vector_updater(self) -> Agent:
        return Agent(
            config=self.agents_config["vector_updater"],  # type: ignore[index]
            tools=[],  # TODO: Add FastAPI upsert tools
            verbose=True,
        )

    @agent
    def index_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["index_manager"],  # type: ignore[index]
            tools=[],  # TODO: Add FastAPI metadata tools
            verbose=True,
        )

    @task
    def update_vectors(self) -> Task:
        return Task(
            config=self.tasks_config["update_vectors"],  # type: ignore[index]
        )

    @task
    def update_index(self) -> Task:
        return Task(
            config=self.tasks_config["update_index"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the UpsertCrew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
