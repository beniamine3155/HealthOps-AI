from fastapi import APIRouter

router = APIRouter()

@router.post("/claim")
def claim_status():
    return {"message": "Claim"}