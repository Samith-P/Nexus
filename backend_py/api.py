from fastapi import FastAPI
from pydantic import BaseModel, Field
from recommender import JournalRecommender

app = FastAPI(
    title="AI-Powered Journal Recommendation API",
    version="1.0.0"
)

recommender = JournalRecommender()

class RecommendRequest(BaseModel):
    abstract: str = Field(..., min_length=30)
    top_k: int = Field(5, ge=1, le=10)

class RecommendResponse(BaseModel):
    recommendations: list

@app.post("/recommend/journals", response_model=RecommendResponse)
def recommend_journals(req: RecommendRequest):
    recs = recommender.recommend(req.abstract, req.top_k)
    return {"recommendations": recs}
