FROM python:3.10

WORKDIR /code

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY out/latest.pkl /model.pkl

COPY app /code/app

ENV MODEL_FILE=/model.pkl
EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
