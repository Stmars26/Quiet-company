web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120 --keep-alive 5 --worker-tmp-dir /dev/shm
worker: python -m publier schedule
