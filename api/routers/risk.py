from fastapi import APIRouter
from api.schemas.risk_schema import RiskPredictionRequest

router = APIRouter()

@router.post("/risk")
def risk_score(request: RiskPredictionRequest):
    return {
            "message": "High", 
            "input": request.model_dump()
        }