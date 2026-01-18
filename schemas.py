from typing import List
from pydantic import BaseModel, Field
class Source(BaseModel):
    url: str = Field(..., description="The URL of the source")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The main content of the agent's response")
    sources: List[Source] = Field(..., description="List of sources referenced in the response")

