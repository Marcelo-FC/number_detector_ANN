# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt . 
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Bootstrap locally via npm
RUN npm install bootstrap@5.3.0

# Ensure the static folder exists and copy Bootstrap files there
RUN mkdir -p /app/static/css /app/static/js && \
    cp node_modules/bootstrap/dist/css/bootstrap.min.css /app/static/css/bootstrap.min.css && \
    cp node_modules/bootstrap/dist/js/bootstrap.bundle.min.js /app/static/js/bootstrap.bundle.min.js

# Copy the rest of the application files
COPY . .

# Expose port 8000 for Django's development server
EXPOSE 8000

# Start the Django application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
