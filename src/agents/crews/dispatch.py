from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List, Optional

from src.agents.tools.qdrant_tools import QdrantSearchTool, QdrantGetNeighborsTool
from src.agents.tools.postgres_schema_inspector import PostgreSQLSchemaInspectorTool
from src.agents.tools.postgres_query_executor import PostgreSQLQueryExecutorTool
from src.llm.ollama import get_llm


@CrewBase
class DispatchCrew:
    """Routes retrieval tasks to the appropriate tools and compiles findings."""

    agents_config = "config/dispatch/agents.yaml"
    tasks_config = "config/dispatch/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, collection_name: Optional[str] = None):
        super().__init__()
        self._collection_name = collection_name

    @agent
    def retrieval_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["retrieval_agent"],  # type: ignore[index]
            tools=[
                QdrantSearchTool(collection_name=self._collection_name),
                QdrantGetNeighborsTool(),
                PostgreSQLSchemaInspectorTool(),
                PostgreSQLQueryExecutorTool(),
            ],
            llm=get_llm(),
            verbose=True,
        )

    @task
    def execute_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["execute_plan_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="DispatchCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
