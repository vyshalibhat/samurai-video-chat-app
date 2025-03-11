
#!/bin/bash

# Kill any existing processes on port 8000
echo "Checking for processes using port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start backend server in the background
echo "Starting FastAPI backend..."
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Give the backend some time to start
sleep 2
echo "Backend started with PID: $BACKEND_PID"

# Go back to project root
cd ..

# Start frontend
echo "Starting React frontend..."
npm start

# When frontend is stopped, stop the backend too
echo "Stopping backend process..."
kill $BACKEND_PID 2>/dev/null || true
