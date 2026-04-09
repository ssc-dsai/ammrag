"""
Pipeline for Summarize flow

Steps:
1. summarize  — produce a comprehensive summary of the input document text
2. condense   — compress the summary to fit within the embedder token budget
"""
import math
import warnings

from dotenv import load_dotenv
from crewai.flow.flow import Flow, listen, start

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from src.agents.crews import SummarizeCrew, CondenseCrew
from src.core.config import settings

# Rough token-to-word ratio for English text (1 token ≈ 0.75 words)
_TOKENS_PER_WORD = 0.75


class SummarizeFlow(Flow):
    model = "gpt-oss"

    @start()
    def summarize(self):
        text = self.state["text"]
        print("Starting Summarize flow — producing full summary")

        crew_output = SummarizeCrew().crew().kickoff(inputs={"text": text})
        summary = str(crew_output)

        self.state["summary"] = summary
        return summary

    @listen(summarize)
    def condense(self, summary: str):
        token_length = self.state.get("token_length") or settings.dense_embedder_token_length
        word_limit = math.floor(token_length * _TOKENS_PER_WORD)

        print(f"Condensing summary to {token_length} tokens (~{word_limit} words)")

        crew_output = CondenseCrew().crew().kickoff(
            inputs={
                "summary": summary,
                "token_length": token_length,
                "word_limit": word_limit,
            }
        )
        condensed = str(crew_output)

        self.state["condensed_summary"] = condensed
        return condensed
