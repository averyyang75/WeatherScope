# WeatherScope Dashboard Service
# FastAPI proxy + static UI for end-to-end workflow demos

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and static assets
COPY app.py .
COPY static ./static

ENV PYTHONUNBUFFERED=1
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
