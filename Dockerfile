FROM python:3.6.4

WORKDIR /apogee

RUN apk add --no-cache gcc musl-dev linux-headers

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["pytest", "tests"]
