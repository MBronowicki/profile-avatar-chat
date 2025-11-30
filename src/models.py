from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

class CacheEntry(BaseModel):
    question: str
    answer: str
    embedding: list[float]