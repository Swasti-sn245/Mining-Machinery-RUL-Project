# MineGuard backend — build from the project ROOT:
#   docker build -t mineguard-backend -f Dockerfile .
#
# This keeps backend/app_v4.py's relative path to ../model/*.pkl working
# unchanged inside the container, exactly as it does locally.

FROM python:3.12-slim

WORKDIR /app

# System deps for scientific Python wheels (xgboost/shap sometimes need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps only (lean image — no matplotlib/jupyter/etc.)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend code and the trained model artifacts
COPY backend/ ./backend/
COPY model/*.pkl model/model_metadata.json ./model/

WORKDIR /app/backend

ENV HOST=0.0.0.0
ENV PORT=5000
EXPOSE 5000

# Healthcheck hits the same /health endpoint the frontend polls
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '5000') + '/health')" || exit 1

# gunicorn + uvicorn workers for production; Render/Railway override CMD via
# their own start command if needed, this is just a sane default.
CMD ["gunicorn", "app_v4:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60"]
