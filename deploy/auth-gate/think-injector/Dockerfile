FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "aiohttp~=3.11"

COPY app.py .

CMD ["python", "-u", "app.py"]
