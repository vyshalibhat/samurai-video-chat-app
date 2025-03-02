#!/bin/bash

# Start backend server in the background
echo "Starting FastAPI backend..."
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Go back to project root
cd ..

# Start frontend
echo "Starting React frontend..."
npm start

# When frontend is stopped, stop the backend too
kill $BACKEND_PID