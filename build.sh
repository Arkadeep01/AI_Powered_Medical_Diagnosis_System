#!/bin/bash

# Build the frontend
echo "Building frontend..."
cd frontend
npm run build
cd ..

echo "Build complete. Run 'python backend/api.py' to start the server."