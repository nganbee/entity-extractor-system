from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

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
# The model is loaded only once when the server starts
print("Initializing NER model...")
ner_model = NERModel()
print("Model initialization complete.")


# --- CORS Configuration ---

# Add CORSMiddleware to allow the Streamlit frontend to call the API
# This allows all origins, methods, and headers, which is suitable for development

origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Only allow origins from the list above
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# --- Pydantic Input/Output Models ---

class TextInput(BaseModel):
    """Defines the expected input format from the client."""
    text: str

class PredictionResponse(BaseModel):
    """Defines the response format sent back to the client."""
    tokens: List[str]
    labels: List[str]


# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the NER API! Visit /docs for documentation."}

@app.post("/predict", response_model=PredictionResponse, tags=["NER"])
def predict(request: TextInput):
    """
    Receives text and returns recognized named entities in BIO format.
    """
    text = request.text
    
    # Tokenize the input text. A simple split by space and punctuation.
    # This is a basic tokenizer. For more complex text, a more sophisticated
    # tokenizer that matches the model's would be better.
    original_tokens = re.findall(r"[\w']+|[.,!?;]", text)
    
    # Initialize all tokens with the "O" (Outside) label
    labels = ["O"] * len(original_tokens)
    
    # Get entity predictions from the model
    entities = ner_model.predict(text)
    
    # A simple mechanism to map character-based entity positions to token-based positions.
    # This works reasonably well but can have edge cases with complex tokenization.
    text_lower = text.lower()
    current_pos = 0
    for token_idx, token in enumerate(original_tokens):
        # Find the start of the token in the text
        token_lower = token.lower()
        start = text_lower.find(token_lower, current_pos)
        if start == -1:
            continue
        end = start + len(token)
        current_pos = end

        # Check if this token is part of any entity
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            
            # Check for overlap between token and entity spans
            if max(start, entity_start) < min(end, entity_end):
                # This token is part of the entity
                entity_label = entity['entity_group']
                
                # Check if it's the first token of the entity
                is_begin = True
                if token_idx > 0:
                    # If the previous label is for the same entity, this is an "Inside" token
                    prev_label = labels[token_idx - 1]
                    if f"I-{entity_label}" == prev_label or f"B-{entity_label}" == prev_label:
                        is_begin = False

                if is_begin:
                    labels[token_idx] = f"B-{entity_label}"
                else:
                    labels[token_idx] = f"I-{entity_label}"
                break # Move to the next token once a label is assigned

    return {"tokens": original_tokens, "labels": labels}
