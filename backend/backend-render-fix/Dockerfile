FROM lokeshgopalan/stock-backend:latest

RUN pip install --no-cache-dir uvicorn[standard] gunicorn

ENV PORT=8000

CMD ["gunicorn","-w","2","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","backend.app:app"]
