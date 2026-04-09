from pydantic import BaseModel
from typing import List


class ParsedQuery(BaseModel):
    orig_query: str
    subquery_list: List[str]
