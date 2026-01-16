#!/bin/bash

# Run script for Crowd Instability Simulation
# Starts both backend and frontend

set -e

echo "=========================================="
echo "Crowd Instability Early-Warning System"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install backend dependencies if needed
if [ ! -d "$SCRIPT_DIR/backend/.venv" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/backend/.venv"
fi

source "$SCRIPT_DIR/backend/.venv/bin/activate"

echo "Installing backend dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# Install frontend dependencies if needed
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd "$SCRIPT_DIR/frontend"
    npm install
fi

# Start backend
echo ""
echo "Starting backend on http://localhost:8000 ..."
cd "$SCRIPT_DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:3000 ..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "System running!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=========================================="

# Handle Ctrl+C
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
