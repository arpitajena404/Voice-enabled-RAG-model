# Interface Documentation

> **For Members 1 (Chunking/Retrieval) and 3 (Generation/Guardrails/Deployment)**
>
> This document describes the typed interfaces, structured models, and entry points
> that Member 2 owns. Use this as the contract when wiring your components into the
> pipeline.

---

## 1. STT Function — `transcribe_audio`

**Module:** `app/stt.py`

```python
async def transcribe_audio(
    file_bytes: bytes,
    filename: str = "audio.webm",
    language_code: str = "hi",
) -> TranscriptionResult:
```

| Parameter       | Type    | Default        | Description |
|-----------------|---------|----------------|-------------|
| `file_bytes`    | `bytes` | *(required)*   | Raw audio bytes (webm, wav, or mp3) |
| `filename`      | `str`   | `"audio.webm"` | Used to infer content-type header |
| `language_code` | `str`   | `"hi"`         | BCP-47 language hint (`hi`, `bn`, `en`) |

**Returns:** `TranscriptionResult` (see §2 below).

**Retry behavior:** Automatically retries on transient failures (HTTP 5xx,
timeouts, connection errors) with exponential backoff. Does NOT retry on
4xx / auth errors. Retry count and backoff are configurable via
`RETRY_MAX_ATTEMPTS`, `RETRY_BACKOFF_BASE`, `RETRY_BACKOFF_MAX` in `.env`.

**Fallback:** When `SARVAM_API_KEY` is not set, returns a simulated
transcript with `used_fallback=True`.

---

## 2. Pydantic Models (`app/schemas.py`)

### `TranscriptionResult`

| Field           | Type    | Description |
|-----------------|---------|-------------|
| `text`          | `str`   | Transcribed text |
| `language_code` | `str`   | Language code used |
| `latency_ms`    | `float` | STT stage wall-clock time (ms) |
| `used_fallback` | `bool`  | True if simulated mode was used |

**Produced by:** STT stage in `app/pipeline.py`

---

### `RetrievalResult`

| Field        | Type         | Description |
|--------------|--------------|-------------|
| `passages`   | `list[dict]` | Ranked passages with `text`, `score`, `dense_score`, `sparse_score`, `url`, `language`, `chunk_type` |
| `strategy`   | `str`        | Chunking strategy used |
| `latency_ms` | `float`      | Retrieval stage wall-clock time (ms) |

**Produced by:** Retrieval stage in `app/pipeline.py`

---

### `GenerationResult`

| Field            | Type         | Description |
|------------------|--------------|-------------|
| `answer`         | `str`        | Final answer text |
| `grounded`       | `bool`       | Grounded in retrieved context? |
| `confidence`     | `float`      | Confidence 0.0–1.0 |
| `citations`      | `list[str]`  | Supporting source URLs |
| `refusal`        | `bool`       | Query was refused? |
| `refusal_reason` | `str`        | Why (empty if not refused) |
| `latency_ms`     | `float`      | Generation stage wall-clock time (ms) |

**Produced by:** Generation + grounding guardrail stages in `app/pipeline.py`

---

### `PipelineResponse`

| Field             | Type                          | Description |
|-------------------|-------------------------------|-------------|
| `transcription`   | `TranscriptionResult \| None` | STT result (None if text input) |
| `retrieval`       | `RetrievalResult \| None`     | Retrieval result |
| `generation`      | `GenerationResult \| None`    | Generation result |
| `query`           | `str`                         | Final query text |
| `answer`          | `str`                         | Final answer |
| `grounded`        | `bool`                        | Overall grounding |
| `confidence`      | `float`                       | Overall confidence |
| `refusal`         | `bool`                        | Was the query refused? |
| `refusal_reason`  | `str`                         | Refusal reason |
| `citations`       | `list[str]`                   | Source citations |
| `passages`        | `list[dict]`                  | Retrieved passages |
| `latencies`       | `dict`                        | `{stt, retrieval, generation, guardrails, total}` in ms |
| `total_latency_ms`| `float`                       | End-to-end wall-clock time (ms) |
| `error`           | `str \| None`                 | Error message (None on success) |

**Method:** `to_api_dict() → dict` — serializes to the flat JSON shape
that `server.py` and `static/app.js` expect (backward compatible).

---

## 3. Pipeline Entry Point — `RAGPipeline.run()`

**Module:** `app/pipeline.py`

```python
class RAGPipeline:
    async def run(
        self,
        query_text: str = None,
        audio_bytes: bytes = None,
        audio_filename: str = "audio.webm",
        language: str = "hi",
        strategy: str = "semantic",
        provider: str = "gemini",
        top_k: int = 3,
    ) -> PipelineResponse:
```

Provide **either** `query_text` (text mode) **or** `audio_bytes` (voice mode).
The pipeline runs STT → Guardrails → Retrieval → Generation → Grounding
and returns a `PipelineResponse`.

A global singleton is exported: `from app.pipeline import pipeline`.

---

## 4. Timing Utility (`app/timing.py`)

```python
from app.timing import timed_stage, timed_stage_sync

# Async
async with timed_stage("stt") as t:
    result = await some_async_call()
print(t.latency_ms)

# Sync
with timed_stage_sync("retrieval") as t:
    result = some_sync_call()
print(t.latency_ms)
```

---

## 5. Running Locally

```bash
# 1. Clone and install
git clone <repo-url>
cd repo
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your SARVAM_API_KEY, GEMINI_API_KEY, etc.

# 3. Seed the database (builds vector indices for all 3 chunking strategies)
python seed_database.py

# 4. Start the server
python main.py
# → http://localhost:8000

# 5. Run latency benchmark
python benchmark_pipeline.py semantic gemini 30 --mode full
# or text-only:
python benchmark_pipeline.py semantic gemini 30 --mode text-only

# 6. Run tests
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

---

## 6. Latency Reports

Reports are written to:
```
data/latency_report_{strategy}.json
```

Where `{strategy}` is `naive`, `semantic`, or `parent_child`.

The report JSON includes:
- `mode` — `"full"` or `"text-only"`
- `latencies` — per-stage `{p50, p70, p100, mean}` for STT, Retrieval,
  Generation, Guardrails, and Total
- `pass_200ms` — boolean pass/fail against 200ms P70 target
- `bottleneck` — name of the stage with highest P70 (when failing)

The `/api/stats?strategy=<name>` endpoint serves these reports to the frontend.

---

## 7. Configuration Reference (`.env`)

| Variable              | Default | Description |
|-----------------------|---------|-------------|
| `SARVAM_API_KEY`      | `""`    | Sarvam AI STT API key |
| `GEMINI_API_KEY`      | `""`    | Google Gemini API key |
| `GROQ_API_KEY`        | `""`    | Groq API key (optional fallback) |
| `RETRY_MAX_ATTEMPTS`  | `3`     | Max retry attempts for transient API failures |
| `RETRY_BACKOFF_BASE`  | `1.0`   | Exponential backoff base (seconds) |
| `RETRY_BACKOFF_MAX`   | `10.0`  | Maximum backoff delay (seconds) |
