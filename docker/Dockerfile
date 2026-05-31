FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2, prophet, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app/

# Fail the image build early if the ASGI module cannot be imported.
RUN python -c "import backend.main; print('backend.main import OK')"

# Ensure data dir exists for Prophet CSV
RUN mkdir -p /app/backend/data

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
