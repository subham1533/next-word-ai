#!/bin/bash
# Run from backend directory if possible, or run backend.main from root
if [ -f "main.py" ]; then
    uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
elif [ -f "backend/main.py" ]; then
    uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
else
    echo "Error: main.py not found!"
    exit 1
fi
