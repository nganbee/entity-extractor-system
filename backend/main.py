from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import the NERModel class from model.py
from core.model import NERModel

# --- App and Model Initialization ---

# Initialize the FastAPI app
app = FastAPI(
    title="Entity Extractor API",
    description="API for Named Entity Recognition using XLM-RoBERTa",
    version="1.0.0"
)

# Create a single, reusable instance of the NERModel
print("Initializing NER model...")
ner_model = NERModel()
print("Model initialization complete.")


# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Input/Output Models ---

class TextInput(BaseModel):
    """Defines the expected input format from the client."""
    text: str

class Entity(BaseModel):
    """Defines the structure of a single entity in the response."""
    entity_group: str
    word: str
    start: int
    end: int


# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the NER API! Visit /docs for documentation."}

@app.post("/predict", response_model=List[Entity], tags=["NER"])
def predict(request: TextInput):
    """
    Receives text and returns a list of recognized named entities.
    The response format is a list of entity objects.
    """
    entities = ner_model.predict(request.text)
    return entities
