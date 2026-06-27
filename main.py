import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_hoja = YOLO("hoja.pt")
model_enfermedades = YOLO("best.pt")

CLASES_ES = {
    "Amarillamiento":    "Amarillamiento",
    "CogolleroAvanzado": "Gusano Cogollero Avanzado",
    "CogolleroInicial":  "Gusano Cogollero Inicial",
    "Mosaico":           "Mosaico",
    "PobredumbreRoja":   "Podredumbre Roja",
    "Roya":              "Roya",
    "Saludable":         "Saludable",
}

@app.get("/")
def root():
    return {"status": "CañaScan API activa"}

@app.get("/clases")
def clases():
    return {"clases": model_enfermedades.names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    resultado_hoja = model_hoja(img_np, imgsz=416, conf=0.45, verbose=False)

    if len(resultado_hoja[0].boxes) == 0:
        return {
            "detecciones": [],
            "imagen_ancho": image.width,
            "imagen_alto": image.height,
            "total": 0,
            "mensaje": "No se detectó hoja de caña en la imagen"
        }

    resultado_enf = model_enfermedades(img_np, imgsz=640, conf=0.15, verbose=False)

    detecciones = []
    for box in resultado_enf[0].boxes:
        clase_en = model_enfermedades.names[int(box.cls)]
        clase_es = CLASES_ES.get(clase_en, clase_en)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detecciones.append({
            "clase":     clase_es,
            "confianza": round(float(box.conf), 4),
            "bbox":      [x1, y1, x2, y2],
        })

    return {
        "detecciones": detecciones,
        "imagen_ancho": image.width,
        "imagen_alto":  image.height,
        "total":        len(detecciones),
        "mensaje":      "Hoja de caña detectada"
    }