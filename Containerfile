# Base image
FROM python:3.12.2-slim-bookworm

# Upgrade all packages to their latest security-patched versions
RUN apt-get update && apt-get install -y git && \
    apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt ./
COPY run.sh ./
COPY conf/ ./conf/
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Make run.sh executable
RUN chmod +x run.sh

# Set entrypoint to allow arg-passing
ENTRYPOINT ["./run.sh"]
