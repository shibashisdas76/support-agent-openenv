FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . /app

# Tell Hugging Face what port to use
EXPOSE 7860

# Start the actual Uvicorn web server!
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]