# H-ANS Hijoluminic Firewall — Containerized Zero-Day Detector
# Build: docker build -t hijoluminic-firewall .
# Run:   docker run -p 8080:8080 hijoluminic-firewall

FROM python:3.11-slim

LABEL maintainer="H-ANS Research"
LABEL description="Hijoluminic Firewall — Zero-Day Network Intrusion Detection via Fractal Resonance Collapse"
LABEL version="1.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy scipy matplotlib \
    fastapi uvicorn \
    scapy \
    psutil

# Expose API port
EXPOSE 8080

# Run the API server
CMD ["python", "-m", "builds.build1_zeroday.api_server"]
