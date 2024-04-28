from fastapi import FastAPI, Request, File, UploadFile 
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi.responses import HTMLResponse


app = FastAPI()

@app.get("/home") 
def home():
    return {"message": "Wtech Yapay Zeka Eğitimi Bitirme Projesi!"}


@app.get("/about") 
def about():
    return {"message": "Melisa YASAK"}


@app.get("/cnn", response_class=HTMLResponse)
async def cnn(request: Request):
    return HTMLResponse(open("C:\\Users\\melis\\WTech_YZE\\bitirmeProjesi_CNN\\cnn.html", "r").read())


@app.get("/ann", response_class=HTMLResponse)
async def ann(request: Request):
    return HTMLResponse(open("C:\\Users\\melis\\WTech_YZE\\bitirmeProjesi_CNN\\ann.html", "r").read())

@app.get("/object_detection", response_class=HTMLResponse)
async def object_detection(request: Request):
    return HTMLResponse(open("C:\\Users\\melis\\WTech_YZE\\bitirmeProjesi_CNN\\object_detection.html", "r").read())

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(open("C:\\Users\\melis\\WTech_YZE\\bitirmeProjesi_CNN\\index.html", "r").read())



@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    load_ann = load_model(r'C:\Users\melis\WTech_YZE\bitirmeProjesi_CNN\image classification with cnn\models\dog-cat2.h5')
    with open("temp_image.jpg", "wb") as f:
        f.write(await file.read())
        
    test_image = Image.open("temp_image.jpg")
    test_image = test_image.resize((240, 240))
    
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0) / 255.0
    
    prediction = load_ann.predict(test_image)
    predicted_class_index = np.argmax(prediction[0])

    class_labels = ["Kedi","Köpek"]

    prediction = class_labels[predicted_class_index]
    return JSONResponse(content={"sonuc": prediction})




