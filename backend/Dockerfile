# Container 1: Model Server (FastAPI + GPU)
# Cài đặt môi trường PyTorch, CUDA, FastAPI
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
