# 1. Gunakan image Python resmi
FROM python:3.11-slim

# 2. Install OS-level dependencies untuk GIS & build tools
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements.txt dan install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. Copy seluruh project ke container
COPY . .

# 6. Jalankan Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
