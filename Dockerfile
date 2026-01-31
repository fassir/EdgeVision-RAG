# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and building tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Fix for potential chromadb/hnswlib build issues on linux
RUN pip install --no-cache-dir pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Rename fallback requirements if necessary or ensure they match
# Note: we are copying the full repo structure
COPY . .

# Expose port
EXPOSE 8000

# Run commands
CMD ["python", "run_api.py"]
