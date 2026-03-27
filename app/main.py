from fastapi import FastAPI
from app.api.routes_ask import router as ask_router
from app.api.routes_ingest import router as ingest_router
from app.core.config import settings

app = FastAPI(title="Asistente Jurídico (MVP)", version="0.1.0")

app.include_router(ask_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.ENV}

@app.get("/")
def root():
    return {"greeting": "Hello world"}