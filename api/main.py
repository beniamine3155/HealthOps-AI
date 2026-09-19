from fastapi import FastAPI
from api.routers import risk, claim, monitoring


app = FastAPI(title="HealthOps ML API")


@app.get("/")
def root():
    return {"message": "HealthOps ML API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(risk.router, prefix="/predict", tags=["Risk Score Prediction"])
app.include_router(claim.router, prefix="/predict", tags=["Claim Status Prediction"])
app.include_router(monitoring.router, prefix="/monitor", tags=["Monitoring"])


print("Registered routes:")
for route in app.routes:
    print(route.path)



