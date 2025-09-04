from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class EvidenceSpan(BaseModel):
    quote: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

class RequirementDebug(BaseModel):
    kw: float
    sim: float
    sim_score: float
    llm: float
    top_sentences: List[EvidenceSpan] = Field(default_factory=list)

class ScoreBreakdown(BaseModel):
    __root__: Dict[str, RequirementDebug]

class AnalysisResult(BaseModel):
    score: float
    verdict: str
    breakdown: Dict[str, RequirementDebug]
    evidence: Dict[str, List[EvidenceSpan]]
    transcript_chars: int
    model_info: Dict[str, str]
