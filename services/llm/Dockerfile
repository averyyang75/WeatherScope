# LLM Weather Interpretation Service
# Supports Ollama (local) and vLLM (cloud) backends

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Environment variables (can be overridden at runtime)
ENV LLM_BACKEND=vllm
ENV OLLAMA_BASE_URL=http://ollama-service:11434
ENV VLLM_BASE_URL=http://host.docker.internal:12434
ENV VLLM_API_PREFIX=/v1
ENV LLM_MODEL=huggingface.co/qwen/qwen3-4b-mlx-4bit:latest
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
