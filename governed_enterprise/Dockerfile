# Use lightweight Python base image
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr (Essential for logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies (Layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose dashboard port
EXPOSE 8080

# Run pre-flight check, then start main engine
CMD python scripts/check_env.py && python gdr_v2_1.py
