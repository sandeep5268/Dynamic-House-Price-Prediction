# Base Python Image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]