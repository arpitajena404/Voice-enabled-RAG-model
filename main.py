import uvicorn

if __name__ == "__main__":
    print("Starting Voice-Enabled Indic RAG Web Server on http://localhost:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

