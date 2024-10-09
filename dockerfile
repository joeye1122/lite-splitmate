# Use the official Python as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirement.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy all project files to the working directory
COPY . .

# Set environment variable to prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Expose the port the application runs on (modify as needed)
EXPOSE 8000

# Command to run the application (modify as needed)
CMD ["python", "FaceRecognizer.py"]
