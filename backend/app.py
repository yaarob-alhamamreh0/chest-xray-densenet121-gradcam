from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.infer import predict_and_cam
import os

app = FastAPI(title="Chest X-Ray Grad-CAM Diagnosis")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    return predict_and_cam(contents)

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
