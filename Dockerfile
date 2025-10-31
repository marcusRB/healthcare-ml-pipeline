FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create a non-root user
RUN useradd -m -u 1001 appuser

# Allow permissions
RUN chown -R appuser:appuser /app

# Switch a non-root user
USER appuser

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]