from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.llm.ollama import get_llm


@CrewBase
class SummarizeCrew:
    """Produces a comprehensive summary of the provided document text."""

    agents_config = "config/summarize/agents.yaml"
    tasks_config = "config/summarize/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],  # type: ignore[index]
            llm=get_llm(),
            verbose=True,
        )

    @task
    def summarize_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="SummarizeCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            tracing=True,
            verbose=True,
        )
