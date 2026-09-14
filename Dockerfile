# Entrega 2: imagen de la API de inferencia.
# Todavia no se usa en la Entrega 1.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
