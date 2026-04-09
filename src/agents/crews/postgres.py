from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.agents.tools.postgres_schema_inspector import PostgreSQLSchemaInspectorTool
from src.agents.tools.postgres_query_executor import PostgreSQLQueryExecutorTool
from src.llm.ollama import get_llm


@CrewBase
class PostgresCrew:
    """Retrieves structured data from PostgreSQL to answer a user query."""

    agents_config = "config/postgres/agents.yaml"
    tasks_config = "config/postgres/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ agents

    @agent
    def schema_inspector(self) -> Agent:
        return Agent(
            config=self.agents_config["schema_inspector"],  # type: ignore[index]
            tools=[PostgreSQLSchemaInspectorTool()],
            llm=get_llm(),
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def sql_query_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["sql_query_developer"],  # type: ignore[index]
            tools=[],
            llm=get_llm(),
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def query_executor(self) -> Agent:
        return Agent(
            config=self.agents_config["query_executor"],  # type: ignore[index]
            tools=[PostgreSQLQueryExecutorTool()],
            llm=get_llm(),
            allow_delegation=False,
            verbose=True,
        )

    # ------------------------------------------------------------------ tasks

    @task
    def inspect_database_schema(self) -> Task:
        return Task(
            config=self.tasks_config["inspect_database_schema"],  # type: ignore[index]
        )

    @task
    def generate_sql_query(self) -> Task:
        return Task(
            config=self.tasks_config["generate_sql_query"],  # type: ignore[index]
        )

    @task
    def execute_query_and_respond(self) -> Task:
        return Task(
            config=self.tasks_config["execute_query_and_respond"],  # type: ignore[index]
        )

    # ------------------------------------------------------------------- crew

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="PostgresCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
