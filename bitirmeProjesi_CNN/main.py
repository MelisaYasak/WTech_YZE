from fastapi import FastAPI
from fastapi import File, UploadFile 
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from tensorflow.keras.models import load_model


app = FastAPI()

@app.get("/home") 
def home():
    return {"message": "Wtech Yapay Zeka Eğitimi Bitirme Projesi!"}


@app.get("/about") 
def about():
    return {"message": "Melisa YASAK"}


@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    load_ann = load_model(r'C:\Users\melis\WTech_YZE\bitirmeProjesi_CNN\image classification with cnn\models\dog-cat2.h5')
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = load_ann.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])

    class_labels = ["Kedi","Köpek"]

    prediction = class_labels[predicted_class_index]
    return {"prediction": prediction}




