# File chạy API server
from fastapi import FastAPI
from core.inference import predict

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def create_prediction(text: str):
    entities = predict(text)
    return {"text": text, "entities": entities}
