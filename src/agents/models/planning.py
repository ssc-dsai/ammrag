from pydantic import BaseModel
from typing import List


class InfoNeed(BaseModel):
    info: str
    query: str


class QueryPlan(BaseModel):
    orig_query: str
    needed_info: List[InfoNeed]
