FROM python:3.11

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app

COPY model/seaAnimal_model3.h5 /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]