# Entity Extractor System

## Abstract

An end-to-end **Named Entity Recognition (NER)** system built on the **XLM-RoBERTa** transformer model, fine-tuned on the [Babelscape/MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd) English subset. The system identifies and classifies named entities across **15 categories** (PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, MYTH, PLANT, TIME, VEHI) from free-form text.

The project is organized into three independent components:

| Component | Purpose | Tech Stack |
|-----------|---------|------------|
| **Modeling** | Fine-tuning, evaluation & pushing model to HF Hub | PyTorch · Transformers · Datasets |
| **Backend** | REST API serving NER predictions | FastAPI · Uvicorn · Transformers Pipeline |
| **Frontend** | Interactive web UI for entity extraction | Streamlit · Requests · Pandas |

---

## Directory Tree

```
entity-extractor-system/
├── .env                          # Environment variables (HF_TOKEN, HF_REPO_ID)
├── .gitignore
├── docker-compose.yml            # Orchestrates backend + frontend containers
├── README.md
│
├── backend/                      # FastAPI model server
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   # API endpoints (/predict, /)
│   └── core/
│
├── frontend/                     # Streamlit web UI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                    # Main Streamlit application
│   └── components/
│
├── modeling/                     # Training & evaluation pipeline
│   ├── requirements.txt
│   ├── config.yml                # Labels, model config & training hyperparams
│   ├── train.py                  # Fine-tuning script
│   ├── evaluation.py             # Batch evaluation on test set
│   ├── data/
│   │   ├── load_data.py          # Download & filter MultiNERD dataset
│   └── notebooks/
│
└── reports/
    └── report_final.pdf          # Project report
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Hugging Face Hub                        │
│              imbee510/finetuned_ner_xlm_roberta             │
└────────────────────┬──────────────────────┬─────────────────┘
                     │  push model          │  pull model
                     │                      │
        ┌────────────┴──────────┐  ┌────────┴────────────────┐
        │      Modeling         │  │       Backend           │
        │  ┌─────────────────┐  │  │  ┌──────────────────┐   │
        │  │  load_data.py   │  │  │  │    NERModel      │   │
        │  │  (MultiNERD)    │  │  │  │  (HF Pipeline)   │   │
        │  └───────┬─────────┘  │  │  └────────┬─────────┘   │
        │          │            │  │           │             │
        │  ┌───────▼─────────┐  │  │  ┌────────▼─────────┐   │
        │  │   train.py      │  │  │  │   FastAPI App    │   │
        │  │ (Fine-tuning)   │  │  │  │   POST /predict  │   │
        │  └───────┬─────────┘  │  │  └────────┬─────────┘   │
        │          │            │  │           │ :8000       │
        │  ┌───────▼─────────┐  │  └───────────┼─────────────┘
        │  │ evaluation.py   │  │              │
        │  │ (seqeval)       │  │              │ HTTP JSON
        │  └─────────────────┘  │              │
        └───────────────────────┘     ┌────────▼─────────────┐
                                      │      Frontend        │
                                      │  ┌────────────────┐  │
                                      │  │  Streamlit App │  │
                                      │  │  (app.py)      │  │
                                      │  └────────────────┘  │
                                      │       :8501          │
                                      └──────────────────────┘
```

**Data Flow:**

1. **Modeling** — Downloads MultiNERD, filters English data, fine-tunes XLM-RoBERTa, evaluates with seqeval, and pushes the model to Hugging Face Hub.
2. **Backend** — Loads the fine-tuned model from Hub, wraps it in a `transformers` pipeline, and exposes a `POST /predict` REST endpoint via FastAPI.
3. **Frontend** — Streamlit web app sends user text to the backend API and renders highlighted entities with color-coded BIO tags.

---

## Environment Setup

### Prerequisites

- **Python** ≥ 3.9
- **Docker** & **Docker Compose** (for containerized deployment)
- **CUDA-capable GPU** (recommended for training; CPU works for inference)

### Environment Variables

Create a `.env` file at the project root:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_REPO_ID=imbee510/finetuned_multiner_xlm_roberta
```

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face access token for pushing models to Hub |
| `HF_REPO_ID` | Target Hugging Face repository ID for model storage |

---

## How to Run

### Docker (Backend + Frontend)

Use Docker Compose to spin up both services with a single command:

```bash
# Build and start containers
docker-compose up --build

# Run in detached mode
docker-compose up --build -d

# Stop all containers
docker-compose down
```

Once running:

| Service | URL |
|---------|-----|
| Backend API | [http://localhost:8000](http://localhost:8000) |
| API Docs (Swagger) | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Frontend UI | [http://localhost:8501](http://localhost:8501) |

> **Note:** The backend container uses `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime` as base image. Ensure your Docker setup supports NVIDIA GPU passthrough if GPU acceleration is needed.

---

### Modeling (Training & Evaluation)

The modeling pipeline runs independently outside Docker.

#### 1. Install Dependencies

```bash
cd modeling

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

#### 2. Download & Prepare Data

```bash
python data/load_data.py
```

This downloads the [Babelscape/MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd) dataset, filters for English samples, and saves the processed splits to `data/processed/multinerd_en/`.

#### 3. Train the Model

```bash
python train.py
```

Training configuration is managed via `config.yml`:

| Parameter | Default |
|-----------|---------|
| Model checkpoint | `FacebookAI/xlm-roberta-base` |
| Max sequence length | `128` |
| Learning rate | `2e-5` |
| Epochs | `3` |
| Batch size | `32` |
| Weight decay | `0.01` |

After training, the model and tokenizer are automatically pushed to Hugging Face Hub using the `HF_REPO_ID` from `.env`.

#### 4. Evaluate the Model

```bash
python evaluation.py
```

Runs batch evaluation on the test split and reports **Precision**, **Recall**, **F1-score**, and **Accuracy** using the `seqeval` metric.
