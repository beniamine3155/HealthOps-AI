from fastapi import APIRouter
from api.schemas.claim_schema import ClaimPredictionRequest

router = APIRouter()

@router.post("/claim")
def claim_status(request: ClaimPredictionRequest):
    return {"message": "Claim", "input": request.model_dump()}