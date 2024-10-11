FROM ultralytics/ultralytics:latest-python

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "YOLOInferenceClass.py"]
