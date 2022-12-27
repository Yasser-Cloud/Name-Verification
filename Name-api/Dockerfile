FROM python:3.9

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY ./nlp /api/nlp

COPY ./storage /api/storage

COPY ./app.py /api/app.py



CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
