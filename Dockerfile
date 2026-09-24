FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable immediate log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for OCR fallback (pytesseract) and container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (skips ~2.5GB of unused CUDA and cuDNN libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy NLP model used by Presidio PII redaction layer
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code (respecting .dockerignore)
COPY . .

# Expose Streamlit default application port
EXPOSE 8080

# Health check to ensure Streamlit server is active and serving requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]