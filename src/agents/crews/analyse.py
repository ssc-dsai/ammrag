import logging
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src.agents.tools.image_analysis_tool import ImageAnalysisTool
from src.agents.tools.placeholder_tool import PlaceholderTool
from src.llm.ollama import get_llm

logger = logging.getLogger(__name__)


@CrewBase
class AnalyseCrew:
    """Per-vector analysis crew — kick off once per retrieved vector.

    Task 1 (vector_refiner): reads stored text, gates the image on relevance,
    classifies steps as text-covered vs needs-tool for images, extracts quotes
    for text chunks. No tools.

    Task 2 (vector_analyst): calls image_analysis tool for each step flagged
    as needs-tool, then assembles the final list of DataPoints.
    """

    agents_config = "config/analyse/agents.yaml"
    tasks_config = "config/analyse/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def vector_refiner(self) -> Agent:
        model = self.agents_config["vector_refiner"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["vector_refiner"],  # type: ignore[index]
            llm=get_llm(model=model),
            max_iter=2,
            verbose=True,
        )

    @agent
    def vector_analyst(self) -> Agent:
        model = self.agents_config["vector_analyst"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["vector_analyst"],  # type: ignore[index]
            llm=get_llm(model=model),
            tools=[ImageAnalysisTool(), PlaceholderTool()],
            max_iter=10,
            verbose=True,
        )

    @task
    def vector_refinement_task(self) -> Task:
        return Task(
            config=self.tasks_config["vector_refinement_task"],  # type: ignore[index]
        )

    @task
    def vector_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["vector_analysis_task"],  # type: ignore[index]
            context=[self.vector_refinement_task()],
            result_as_answer=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="AnalyseCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
