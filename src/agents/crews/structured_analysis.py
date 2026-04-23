import logging
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src.llm.ollama import get_llm

logger = logging.getLogger(__name__)


@CrewBase
class StructuredAnalysisCrew:
    """Placeholder crew for structured file analysis — not yet implemented."""

    agents_config = "config/structured_analysis/agents.yaml"
    tasks_config = "config/structured_analysis/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def structured_analyst(self) -> Agent:
        model = self.agents_config["structured_analyst"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["structured_analyst"],  # type: ignore[index]
            llm=get_llm(model=model),
            max_iter=1,
            verbose=False,
        )

    @task
    def structured_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["structured_analysis_task"],  # type: ignore[index]
            result_as_answer=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="StructuredAnalysisCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
