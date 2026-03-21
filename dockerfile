FROM python:3.11-slim

# Install FFmpeg (required by the app for audio export)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/Gegbee/Natural-Ambience-Synthesizer.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Run the Flask app, binding to all interfaces so it's reachable outside the container
CMD ["python", "-c", "import app; app.app.run(host='0.0.0.0', port=5000)"]