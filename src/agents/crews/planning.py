from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
import crewai.agents.parser as _parser_module
from crewai.agents.parser import AgentFinish
from crewai.agents.constants import FINAL_ANSWER_ACTION
from crewai.project import CrewBase, agent, crew, task
from typing import List

from src.agents.models.planning import QueryPlan
from src.llm.ollama import get_llm

# Patch: when the LLM outputs raw JSON (no tools, no ReAct wrapper), accept it as
# a Final Answer instead of raising a format error and retrying.
_orig_parse = _parser_module.parse

def _patched_parse(text: str):
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}") and FINAL_ANSWER_ACTION not in text:
        return AgentFinish(thought="", output=stripped, text=text)
    return _orig_parse(text)

_parser_module.parse = _patched_parse  # type: ignore[assignment]


@CrewBase
class PlanningCrew:
    """Decomposes a user question into discrete information needs."""

    agents_config = "config/planning/agents.yaml"
    tasks_config = "config/planning/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def query_expert(self) -> Agent:
        model = self.agents_config["query_expert"].get("llm")  # type: ignore[index]
        return Agent(
            config=self.agents_config["query_expert"],  # type: ignore[index]
            llm=get_llm(model=model),
            max_iter=2,
            verbose=True,
        )

    @task
    def query_decomposition_task(self) -> Task:
        def _set_pydantic(output):
            try:
                raw = output.raw or ""
                if "```" in raw:
                    raw = raw.split("```")[-2].lstrip("json").strip()
                output.pydantic = QueryPlan.model_validate_json(raw.strip())
            except Exception:
                pass

        return Task(
            config=self.tasks_config["query_decomposition_task"],  # type: ignore[index]
            result_as_answer=True,
            callback=_set_pydantic,
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            name="PlanningCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Get the floorplans for all offices, then find the kitchen location in each one."
    print(f"Query: {query}\n")

    PlanningCrew().crew().kickoff(inputs={"query": query})