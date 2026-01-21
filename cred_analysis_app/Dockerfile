FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directory exists
RUN mkdir -p data

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
