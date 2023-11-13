import base64
from imp import load_module
from io import BytesIO
from tkinter import Image
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = FastAPI()

def find_file(file_name, search_path):
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file == file_name:
                return os.path.join(root, file)
    return None

model_file_path = find_file("seaAnimal_model3.h5", "/app")

if model_file_path:
    model = keras.models.load_model(model_file_path)
    print(model.summary())
else:
    print("Model file not found")

class_labels = {
    0: 'Dolphin',
    1: 'Fish',
    2: 'Lobster',
    3: 'SeaUrchins',
    4: 'Starfish',
    5: 'TurtleTortoise'
}

@app.post("/predict/")
async def preprocess_and_predict(request: Request):
    try:
        data = await request.json()
        img_base64 = data.get("img")

        if img_base64:

            result = model.predict(np.array([img_base64]))
            predicted_class_index = np.argmax(result)
            predicted_class_label = class_labels[predicted_class_index]

            return JSONResponse(content={"ผลลัพธ์ที่ได้": predicted_class_label})
        else:
            return JSONResponse(content={"error": "No 'img' provided in request"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
@app.get("/")
def root():
    return {"this is my API"}
