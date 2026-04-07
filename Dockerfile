FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install (This breaks the cache so it actually installs!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . /app

# Tell Hugging Face what port to use
EXPOSE 7860

# Start the server on the correct Hugging Face port
CMD ["python", "-m", "openenv.server", "--port", "7860", "--host", "0.0.0.0"]