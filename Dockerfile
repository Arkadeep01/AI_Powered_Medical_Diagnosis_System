# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirement.txt ./
RUN pip install --no-cache-dir -r requirement.txt

# Copy backend code
COPY backend/ ./backend/

# Copy pre-built frontend (build locally first with ./build.sh)
COPY frontend/dist ./frontend/dist

# Copy model files
COPY SAV_File/ ./SAV_File/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "backend/api.py"]