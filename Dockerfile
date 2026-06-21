FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY DSen2-CR ./DSen2-CR

WORKDIR /app/DSen2-CR

CMD ["python", "test.py"]