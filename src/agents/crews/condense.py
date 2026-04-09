from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.llm.ollama import get_llm


@CrewBase
class CondenseCrew:
    """Condenses a summary to fit within the embedder token limit."""

    agents_config = "config/condense/agents.yaml"
    tasks_config = "config/condense/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def condenser(self) -> Agent:
        return Agent(
            config=self.agents_config["condenser"],  # type: ignore[index]
            llm=get_llm(),
            verbose=True,
        )

    @task
    def condense_task(self) -> Task:
        return Task(
            config=self.tasks_config["condense_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="CondenseCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            tracing=True,
            verbose=True,
        )
