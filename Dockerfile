FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version for lightweight, fast container execution
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files including LFS-tracked models and processed graphs
COPY . /app

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false"]