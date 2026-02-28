FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than CUDA version)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    # Install remaining deps, skipping torch so pip doesn't pull CUDA version
    grep -vi '^torch$' requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir --prefix=/install -r requirements_no_torch.txt

# ---------- Final lightweight stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed Python packages from the builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
