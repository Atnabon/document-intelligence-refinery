FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY rubric/ rubric/
COPY main.py ./
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Create output directories
RUN mkdir -p .refinery/profiles .refinery/pageindex .refinery/vectorstore data/raw

# Default entry point
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
