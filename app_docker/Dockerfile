FROM python:3.11-slim

# System dependencies for PySCF and DScribe
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from the root requirements.txt
# (passed in via docker-compose build context which is the repo root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend (paths relative to repo root, which is the build context)
COPY app/backend/main.py .
COPY app/backend/inference.py .

# Copy frontend
COPY app/frontend/ ./frontend/

# Mount point for user's project files (deephf_electronic.py etc.)
# These are bind-mounted at runtime via docker-compose, not baked in.
# This allows the user to update their code without rebuilding.
ENV PYTHONPATH="/app/project:${PYTHONPATH}"

# Cache directory persisted via volume
RUN mkdir -p /app/cache_electronic

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
