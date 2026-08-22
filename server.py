import os
import json
import logging
import pickle
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.config import config
from app.pipeline import pipeline
from app.indexer import indexer
import seed_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="Voice-Enabled Indic RAG API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text query request model
class TextQueryRequest(BaseModel):
    query: str
    language: str = "hi"
    strategy: str = "semantic"
    provider: str = "gemini"
    top_k: int = 3

@app.on_event("startup")
def startup_event():
    """Verify index status and pre-warm embedding model on startup."""
    logger.info("Server starting up...")
    # Attempt to load indexes for semantic strategy to check if seeded
    semantic_dense = config.INDEX_DIR / "semantic_dense.npy"
    if not semantic_dense.exists():
        logger.warning("No indices found. Seeding with a tiny default sample...")
        try:
            seed_database.seed(limit_per_language=10)
        except Exception as e:
            logger.error(f"Auto-seeding failed: {e}. Index will need to be generated manually.")
    
    # Pre-warm embedding model and index in RAM for sub-100ms first query latency
    try:
        logger.info("Pre-warming sentence-transformer embedding model...")
        indexer.get_embedding_model()
        indexer.indices.clear()
        indexer.get_index("semantic", force_reload=True)
        logger.info("Embedding model and indices pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Embedding model pre-warming warning: {e}")



@app.post("/api/query")
async def handle_query(
    query_text: str = Form(None),
    language: str = Form("hi"),
    strategy: str = Form("semantic"),
    provider: str = Form("gemini"),
    top_k: int = Form(3),
    audio_file: UploadFile = File(None)
):
    """
    Handles end-to-end RAG query. Supports text input or audio file input.
    """
    try:
        audio_bytes = None
        filename = "audio.webm"
        
        if audio_file:
            audio_bytes = await audio_file.read()
            filename = audio_file.filename
            logger.info(f"Received audio file {filename} ({len(audio_bytes)} bytes)")
            
        # Run pipeline
        res = await pipeline.run(
            query_text=query_text,
            audio_bytes=audio_bytes,
            audio_filename=filename,
            language=language,
            strategy=strategy,
            provider=provider,
            top_k=int(top_k)
        )
        # Serialize to the dict shape the frontend (app.js) expects
        return res.to_api_dict()
        
    except Exception as e:
        logger.exception("Error handling pipeline query")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/languages")
def get_languages():
    """Returns supported languages."""
    return [
        {"code": "hi", "name": "Hindi (हिन्दी)"},
        {"code": "bn", "name": "Bengali (বাংলা)"},
        {"code": "en", "name": "English"},
        {"code": "ta", "name": "Tamil (தமிழ்)"},
        {"code": "te", "name": "Telugu (తెలుగు)"},
        {"code": "mr", "name": "Marathi (मराठी)"},
        {"code": "gu", "name": "Gujarati (ગુજરાતી)"},
        {"code": "kn", "name": "Kannada (ಕನ್ನಡ)"},
        {"code": "ml", "name": "Malayalam (മലയാളം)"},
        {"code": "pa", "name": "Punjabi (ਪੰਜਾਬੀ)"},
        {"code": "od", "name": "Odia (ଓଡ଼ਿଆ)"}
    ]


@app.get("/api/stats")
def get_stats(strategy: str = "semantic"):
    """
    Loads and returns latency reports from disk if they exist.
    """
    report_file = config.DATA_DIR / f"latency_report_{strategy}.json"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": f"No benchmark stats found for strategy {strategy}. Run benchmark first."}

@app.post("/api/seed")
def trigger_seed(limit: int = 100):
    """Triggers database indexing manually."""
    try:
        indexer.indices.clear()
        seed_database.seed(limit)
        return {"status": "success", "message": "Database seeded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seeding failed: {str(e)}")


# Mount static folder for the frontend dashboard
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning("Static assets directory not found. Frontend will not be served.")
