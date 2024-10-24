#!/bin/bash

echo "Starting app.py..."
python3 app.py &

echo "Starting YOLOInferenceClass.py..."
python3 YOLOInferenceClass.py & 
wait

echo "Splitmate have been started."
