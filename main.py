import io
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

# Cargar modelo
model = YOLO("best.pt")

# Nombres de clases — 7 clases (ModeloDemo_v5)
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
    return {"clases": model.names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image, imgsz=640, conf=0.25)
    result = results[0]

    detecciones = []
    for box in result.boxes:
        clase_en = model.names[int(box.cls)]
        clase_es = CLASES_ES.get(clase_en, clase_en)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detecciones.append({
            "clase":     clase_es,
            "confianza": round(float(box.conf), 4),
            "bbox":      [x1, y1, x2, y2],
        })

    ancho, alto = image.size

    return {
        "detecciones": detecciones,
        "imagen_ancho": ancho,
        "imagen_alto":  alto,
        "total":        len(detecciones),
    }
