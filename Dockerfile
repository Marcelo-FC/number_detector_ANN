FROM python:3.9

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# ✅ Ensure the static folder exists
RUN mkdir -p /app/static

# ✅ Force TensorFlow to use CPU only
ENV CUDA_VISIBLE_DEVICES="-1"

# Expose port
EXPOSE 8000

# Start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
