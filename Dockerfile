# Entrega 2: imagen de la API de inferencia.
# Antes de construirla tiene que existir el modelo:
#   python -m src.training.train
FROM python:3.11-slim

WORKDIR /app

# primero las dependencias (si no cambia requirements, docker no las reinstala)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# codigo de la API y de la inferencia
COPY app/ app/
COPY src/ src/

# el modelo entrenado va adentro de la imagen
COPY models/ models/

EXPOSE 8000

# sin --reload (eso es solo para desarrollo)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
