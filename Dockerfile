FROM python:3.12.8-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:7860", "main:app"]

