# Multi-stage lightweight Python runtime with Java for Apache Spark
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install headless Java JRE for PySpark and curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java home environment variable
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit server readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
