FROM python:3.13.7

WORKDIR /rag_vardhan


COPY requirements.txt

RUN pip install --nocache-dir -r requirements.txt


COPY . .

EXPOSE 8010

CMD ["uvicorn", "main2:app", "--host", "0.0.0.0"]
