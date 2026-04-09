from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.agents.models.parsing import ParsedQuery
from src.llm.ollama import get_llm

@CrewBase
class ParsingCrew:
    """Parses complex queries into sub-queries, retrieves relevant context, and synthesizes answers."""

    agents_config = "config/parsing/agents.yaml"
    tasks_config = "config/parsing/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ agents

    @agent
    def query_expert(self) -> Agent:
        model = self.agents_config["query_expert"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["query_expert"],  # type: ignore[index]
            llm=get_llm(model=model),
            verbose=True,
        )

    # ------------------------------------------------------------------ tasks

    @task
    def query_decomposition_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_decomposition_task"],  # type: ignore[index]
            result_as_answer=True,
            output_pydantic=ParsedQuery,
            
        )

    # ------------------------------------------------------------------- crew

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="ParsingCrew",
            agents=self.agents,  
            tasks=self.tasks,    
            process=Process.sequential,
            share_crew=True,
            verbose=True,
        )

