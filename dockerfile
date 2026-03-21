FROM python:3.11-slim

# Install FFmpeg and PortAudio (required by soundfile and sounddevice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application source
COPY . .

# Render injects PORT at runtime; default to 10000 (Render's default)
ENV PORT=10000

EXPOSE 10000

# Use gunicorn for production, do NOT use Flask's built-in dev server
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 2 app:app
