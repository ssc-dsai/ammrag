from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.agents.models.answer import QueryAnswer
from src.llm.ollama import get_llm

@CrewBase
class FormatCrew:
    """Parses complex queries into sub-queries, retrieves relevant context, and synthesizes answers."""

    agents_config = "config/format/agents.yaml"
    tasks_config = "config/format/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ agents

    @agent
    def answer_synthesizer(self) -> Agent:
        model = self.agents_config["answer_synthesizer"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["answer_synthesizer"],  # type: ignore[index]
            llm=get_llm(model=model),
            verbose=True,
        )

    # ------------------------------------------------------------------ tasks

    @task
    def answer_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_synthesis_task"],  # type: ignore[index]
        )

    # ------------------------------------------------------------------- crew

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="FormatCrew",
            agents=self.agents,  
            tasks=self.tasks,    
            process=Process.sequential,
            verbose=True,
        )

