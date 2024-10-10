# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install cmake
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# Copy the local requirements.txt to the container
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# If there are other source code files, you can also copy them to the container
# COPY . .

# Set the default command to run when starting the container
# Replace this with your application's startup command, e.g., python app.py
CMD ["python", "--version"]
