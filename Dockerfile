FROM python:3.10-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . /app

# Tell Hugging Face what port to use
EXPOSE 7860

# Start the OpenEnv server and point it to your environment
CMD ["python", "-m", "openenv.server", "server/env.py", "--port", "7860", "--host", "0.0.0.0"]