import os
import io
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------
# APP
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# RUTAS DE MODELOS (backend/)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

modelrecorte = YOLO(os.path.join(BACKEND_DIR, "recorte2.pt"))
modeldetOP = YOLO(os.path.join(BACKEND_DIR, "3clsOPfft.pt"))
modeldetOA = YOLO(os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt"))

# ---------------------------
# FUNCIONES ORIGINALES (SIN CAMBIOS DE LÓGICA)
# ---------------------------

def yolorecorte(model, img):
    results = model(img)
    coor = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor

def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = model(ms)
    cls = int(results[0].probs.top1)
    prob = float(results[0].probs.top1conf)
    return cls, prob

def yolodetOA(model, crop, certeza=0):
    results = model(crop)
    cls = []
    prob = []
    coords = []

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append(tuple(map(int, box.xyxy[0])))

    if not prob:
        return None

    idx = int(np.argmax(prob))
    x1, y1, x2, y2 = coords[idx]
    return cls[idx], prob[idx], x1, y1, x2, y2

def etiquetar2(imagen, clOP, xOP1, yOP1, xOP2, yOP2, clOA, xOA1, yOA1, xOA2, yOA2):
    img = imagen.copy()

    cv2.rectangle(img, (xOP1, yOP1), (xOP2, yOP2), (255, 0, 0), 2)

    if clOP == 0:
        etiqueta = "Sin osteoporosis"
    elif clOP == 1:
        etiqueta = "Osteopenia"
    else:
        etiqueta = "Osteoporosis"

    cv2.putText(img, etiqueta, (xOP1, yOP1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    p1OA = (xOP1 + xOA1, yOP1 + yOA1)
    p2OA = (xOP1 + xOA2, yOP1 + yOA2)

    cv2.rectangle(img, p1OA, p2OA, (0, 0, 255), 2)

    etiquetas_oa = [
        "Sin Osteoartrosis",
        "OA dudoso",
        "OA leve",
        "OA moderado",
        "OA grave"
    ]

    cv2.putText(img, etiquetas_oa[clOA],
                (p1OA[0], p1OA[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

# ---------------------------
# PIPELINE (MISMA LÓGICA, ORDENADA)
# ---------------------------
def CorrerModelo(img):
    coor = yolorecorte(modelrecorte, img)

    if not coor:
        raise HTTPException(status_code=400, detail="No se detectó rodilla")

    crops = []
    etiquetadas = []

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]].copy()
        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        oa = yolodetOA(modeldetOA, crop)

        crops.append(crop)

        if oa:
            clOA, _, x1, y1, x2, y2 = oa
            etiquetada = etiquetar2(
                img, clOP,
                c[0], c[1], c[2], c[3],
                clOA, x1, y1, x2, y2
            )
            etiquetadas.append(etiquetada)

    # imagen procesada = solo recortes (1 o 2)
    imagen_procesada = np.hstack(crops) if len(crops) > 1 else crops[0]

    # imagen etiquetada = última etiquetada (como en tu flujo)
    imagen_etiquetada = etiquetadas[-1] if etiquetadas else imagen_procesada

    return imagen_procesada, imagen_etiquetada

# ---------------------------
# API
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    procesada, etiquetada = CorrerModelo(img)

    def to_base64(im):
        _, buffer = cv2.imencode(".jpg", im)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    return {
        "mensaje":"Ánalisis completo",
        "imagenProcesada": to_base64(procesada),
        "imagenEtiquetada": to_base64(etiquetada)
    }

@app.get("/")
def health():
    return {"status": "ok"}
