FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]