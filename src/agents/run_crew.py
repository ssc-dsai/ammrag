"""
Crew runner - standalone entry point for running crews with proper import paths.
Called by the crew service. Outputs JSON to stdout for the caller to parse.
"""

import sys
import os
import json
import warnings
import logging

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Ensure the agents directory is on the path so nss_agents imports resolve
agents_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, agents_dir)

# Register nss_agents as a module alias pointing to this directory
import importlib
import types

nss_pkg = types.ModuleType("nss_agents")
nss_pkg.__path__ = [agents_dir]
sys.modules["nss_agents"] = nss_pkg

for mod_name in ["models", "config", "storage"]:
    full = f"nss_agents.{mod_name}"
    sys.modules[full] = importlib.import_module(mod_name)

# Tools sub-package
tools_pkg = types.ModuleType("nss_agents.tools")
tools_pkg.__path__ = [os.path.join(agents_dir, "tools")]
sys.modules["nss_agents.tools"] = tools_pkg

for mod_name in ["tools.sql_tool", "tools.qdrant_tool", "tools.qdrant_tool.tool"]:
    full = f"nss_agents.{mod_name}"
    sys.modules[full] = importlib.import_module(mod_name)

# Now import the crews - NssAgents and KnowledgeOrganizingCrew work directly.
# RetrievalCrew has a metaclass conflict with BaseCrew(abc.ABC) + @CrewBase.
# We import what we can and handle errors gracefully.
AVAILABLE_CREWS = {}

try:
    from crew import NssAgents
    AVAILABLE_CREWS["research"] = NssAgents
except Exception as e:
    logging.warning(f"Could not import NssAgents: {e}")

try:
    from crew import KnowledgeOrganizingCrew
    AVAILABLE_CREWS["knowledge"] = KnowledgeOrganizingCrew
except Exception as e:
    logging.warning(f"Could not import KnowledgeOrganizingCrew: {e}")

try:
    from crew import RetrievalCrew
    AVAILABLE_CREWS["retrieval"] = RetrievalCrew
except Exception as e:
    logging.warning(f"Could not import RetrievalCrew: {e}")


# Crews that inherit BaseCrew need config arguments
CREWS_NEEDING_CONFIG = {"retrieval", "knowledge"}


def _instantiate_crew(crew_type: str):
    """Instantiate a crew, passing BaseCrew config if needed."""
    import config as agent_config

    crew_cls = AVAILABLE_CREWS[crew_type]

    if crew_type in CREWS_NEEDING_CONFIG:
        return crew_cls(
            embedder_config=agent_config.embedder_config,
            qdrant_location=agent_config.qdrant_location,
            qdrant_api_key=agent_config.qdrant_api_key,
        )
    return crew_cls()


def run_crew(crew_type: str, inputs: dict) -> dict:
    """Run a crew and return the result as a dict."""
    if crew_type not in AVAILABLE_CREWS:
        return {
            "status": "error",
            "crew_type": crew_type,
            "error": f"Crew '{crew_type}' not available. "
                     f"Available: {list(AVAILABLE_CREWS.keys())}"
        }

    try:
        crew_instance = _instantiate_crew(crew_type)
        result = crew_instance.crew().kickoff(inputs=inputs)

        return {
            "status": "completed",
            "crew_type": crew_type,
            "result": str(result)
        }
    except Exception as e:
        return {
            "status": "error",
            "crew_type": crew_type,
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Usage: python run_crew.py <crew_type> <json_inputs>
    Example: python run_crew.py research '{"topic": "AI LLMs", "current_year": "2026"}'
    """
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "error": "Usage: python run_crew.py <crew_type> [json_inputs]"
        }))
        sys.exit(1)

    crew_type = sys.argv[1]

    if crew_type == "--list":
        print(json.dumps({
            "status": "ok",
            "available_crews": list(AVAILABLE_CREWS.keys())
        }))
        sys.exit(0)

    inputs = {}
    if len(sys.argv) >= 3:
        try:
            inputs = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print(json.dumps({
                "status": "error",
                "error": "Invalid JSON inputs"
            }))
            sys.exit(1)

    result = run_crew(crew_type, inputs)
    print(json.dumps(result))
