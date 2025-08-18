FROM lokeshgopalan/stock-backend:latest

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

RUN pip install --no-cache-dir uvicorn[standard]==0.30.1 gunicorn==23.0.0

EXPOSE 8000

CMD ["gunicorn","-w","2","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","backend.app:app"]
