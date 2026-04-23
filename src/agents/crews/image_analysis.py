import logging
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src.agents.tools.image_analysis_tool import ImageAnalysisTool
from src.llm.ollama import get_llm

logger = logging.getLogger(__name__)


@CrewBase
class ImageAnalysisCrew:
    """Batch image analysis crew.

    Task 1 (image_gate_agent): reads stored text descriptions for all images
    and gates out irrelevant ones without calling any tools.

    Task 2 (image_analysis_agent): for each image that passed the gate,
    calls image_analysis tool once per step the text did not cover.
    Produces the final DataPoint list.
    """

    agents_config = "config/image_analysis/agents.yaml"
    tasks_config = "config/image_analysis/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def image_gate_agent(self) -> Agent:
        model = self.agents_config["image_gate_agent"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["image_gate_agent"],  # type: ignore[index]
            llm=get_llm(model=model),
            max_iter=2,
            verbose=True,
        )

    @agent
    def image_analysis_agent(self) -> Agent:
        model = self.agents_config["image_analysis_agent"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["image_analysis_agent"],  # type: ignore[index]
            llm=get_llm(model=model),
            tools=[ImageAnalysisTool()],
            max_iter=20,
            verbose=True,
        )

    @task
    def image_gate_task(self) -> Task:
        return Task(
            config=self.tasks_config["image_gate_task"],  # type: ignore[index]
        )

    @task
    def image_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["image_analysis_task"],  # type: ignore[index]
            context=[self.image_gate_task()],
            result_as_answer=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="ImageAnalysisCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
