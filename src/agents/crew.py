import os
from dotenv import load_dotenv

from src.core.config import settings

load_dotenv()

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.agents.tools.qdrant_tools import (
    QdrantSearchTool,
    QdrantGetNeighborsTool,
    QdrantGetDocumentTool,
)

_llm = LLM(
    model=os.environ.get("MODEL", "openai/llama3.2:latest"),
    base_url=f"{settings.ollama_host.rstrip('/')}/v1",
    api_key="ollama",
)

# def _llm() -> LLM:
#     # Ollama
#     model = os.environ.get("MODEL", "ollama/gpt-oss")
#     api_base = os.environ.get("API_BASE", "http://localhost:11434").rstrip("/")
#     return LLM(
#         model=model,
#         provider="openai",
#         base_url=f"{api_base}/v1",
#         api_key="ollama",
#     )


@CrewBase
class AmmRagCrew:
    """AMM RAG Agents — three-agent retrieval-augmented generation crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ agents

    @agent
    def question_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["question_expert"],  # type: ignore[index]
            tools=[QdrantSearchTool()],
            llm=_llm,
            verbose=True,
        )

    @agent
    def context_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["context_specialist"],  # type: ignore[index]
            tools=[QdrantGetNeighborsTool(), QdrantGetDocumentTool()],
            llm=_llm,
            verbose=True,
            max_iter=30,
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],  # type: ignore[index]
            llm=_llm,
            verbose=True,
        )

    # ------------------------------------------------------------------ tasks

    @task
    def query_decomposition_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_decomposition_task"],  # type: ignore[index]
            result_as_answer=True,                                 # type: ignore[index]
        )

    @task
    def query_retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_retrieval_task"],  # type: ignore[index]
            result_as_answer=True,                             # type: ignore[index]
        )

    @task
    def context_gathering_task(self) -> Task:
        return Task(
            config=self.tasks_config["context_gathering_task"],  # type: ignore[index]
        )

    @task
    def answer_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_synthesis_task"],  # type: ignore[index]
        )

    # ------------------------------------------------------------------- crew

    @crew
    def crew(self) -> Crew:
        """Sequential RAG crew: retrieve → expand context → synthesise answer."""
        return Crew(
            agents=self.agents,  
            tasks=self.tasks,    
            process=Process.sequential,
            verbose=True,
        )

