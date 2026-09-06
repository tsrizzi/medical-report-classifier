from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto do laudo medico a ser triado")


class TriageResponse(BaseModel):
    label: str
    scores: dict[str, float]
