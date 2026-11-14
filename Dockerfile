FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so Docker caches dependencies cleanly
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application Python modules
COPY main.py slicer.py excel_builder.py gemini_client.py ./

# Render sets the PORT environment variable automatically
CMD ["sh", "-c", "export GOOGLE_SERVICE_ACCOUNT_JSON=\"$(cat /etc/secrets/sa.json)\" && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

