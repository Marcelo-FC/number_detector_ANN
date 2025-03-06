# Use the slim version of Python 3.9
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies manually
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the static folder exists
RUN mkdir -p /app/static

# Copy the rest of the application files
COPY . .

# Expose port 8000 for Django
EXPOSE 8000

# Start Django application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
