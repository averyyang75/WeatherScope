# WeatherScope Map Service
# Uses conda-forge Cartopy packages to avoid flaky apt mirror issues on arm64.

FROM condaforge/miniforge3:24.11.3-0

WORKDIR /app

# Install Cartopy stack from conda-forge (reliable on linux/arm64),
# then install app dependencies from pip requirements.
RUN mamba install -y -n base \
    python=3.11 \
    cartopy=0.23.* \
    && mamba clean -afy

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
