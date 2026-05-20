# Container 1: Model Server (FastAPI + CPU)
# Cài đặt môi trường Python nhẹ và PyTorch bản CPU
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Cài đặt các thư viện, chỉ định tải PyTorch bản CPU để giảm dung lượng
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
