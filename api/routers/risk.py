from fastapi import APIRouter

router = APIRouter()

@router.post("/risk")
def risk_score():
    return {"message": "High"}