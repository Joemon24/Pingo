from fastapi import FastAPI

app = FastAPI(title="pingo")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}