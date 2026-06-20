FROM --platform=linux/amd64 python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/data/soros.db

CMD ["python", "main.py"]
