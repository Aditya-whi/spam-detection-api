from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}

@app.post("/predict")
def predict(data: TextRequest):
    transformed = vectorizer.transform([data.text])
    prediction = model.predict(transformed)
    
    if prediction[0] == 1:
        return {"prediction": "Spam"}
    else:
        return {"prediction": "Ham"}