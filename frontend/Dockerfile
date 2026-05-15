# Container 2: Web UI (Streamlit)
# Cài đặt môi trường Python nhẹ cho UI
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
