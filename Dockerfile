# ─────────────────────────────────────────────
#  GestureTalk – Dockerfile
#  Base: Python 3.10 (matches runtime.txt)
# ─────────────────────────────────────────────
FROM python:3.10.13-slim

# --- System dependencies ---
# libgl1 + libglib2.0 are required by OpenCV
# libportaudio2 is required by pyttsx3 / audio libs
# java is required by language-tool-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libportaudio2 \
    default-jre-headless \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install all Python packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY . .

# --- Expose Flask port ---
EXPOSE 5000

# --- Environment variables ---
# Set Python output to unbuffered so logs appear in real time
ENV PYTHONUNBUFFERED=1
# Prevent pyttsx3 from trying to open a display
ENV DISPLAY=:99

# --- Entrypoint ---
CMD ["python", "app.py"]
