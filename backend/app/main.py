
from fastapi import FastAPI

app = FastAPI(
    title="Backend API",
    description="A starter FastAPI backend",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
