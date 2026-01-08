import os
import cv2
import numpy as np
import base64
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


# -------------------------------------------------
# 0) CONFIG FASTAPI + CORS
# -------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# 1) CARGAR MODELOS SOLO UNA VEZ
# -------------------------------------------------

base_path = os.path.dirname(__file__)

path_detOP = os.path.join(base_path, "backend", "det 2cls R2 0.pt")
path_detOA = os.path.join(base_path, "backend", "OAyoloIR4AH.pt")


@lru_cache(maxsize=1)
def load_models():
    print(">>> [MODELOS] Cargando modelos YOLO...")
    modeldetOP = YOLO(path_detOP)
    modeldetOA = YOLO(path_detOA)
    print(">>> [MODELOS] Modelos cargados OK")
    return modeldetOP, modeldetOA


# PRECARGAR MODELOS CUANDO INICIA EL SERVIDOR (EVITA 502)
@app.on_event("startup")
def preload():
    print(">>> Precargando modelos en memoria...")
    load_models()


# -------------------------------------------------
# 2) FUNCIONES DE PROCESAMIENTO
# -------------------------------------------------

def yolodetOPCrop(modeldet, img, certeza):
    results = modeldet(img)
    cls, prob, crops, coords = [], [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            conf = box.conf[0].item()

            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                crops.append(crop)
                coords.append((x1, y1, x2, y2))

    if not prob:
        return 0, 0, None, 0, 0, 0, 0

    x = prob.index(max(prob))
    return cls[x], prob[x], crops[x], *coords[x]


def yolodetOA(modeldet, img, certeza):
    results = modeldet(img)
    cls, prob, coords = [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append((x1, y1, x2, y2))

    if not prob:
        return 0, 0, 0, 0, 0, 0

    x = prob.index(max(prob))
    return cls[x], prob[x], *coords[x]


def etiquetar(img, clOP, xOP1, yOP1, xOP2, yOP2, clOA, xOA1, yOA1, xOA2, yOA2):
    color = (255, 0, 0)
    grosor = 2

    cv2.rectangle(img, (xOP1, yOP1), (xOP2, yOP2), color, grosor)
    etiqueta = "normal" if clOP == 0 else "osteoporosis"
    cv2.putText(img, etiqueta, (xOP1 + 20, yOP1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    p1 = (xOP1 + xOA1, yOP1 + yOA1)
    p2 = (xOP1 + xOA2, yOP1 + yOA2)
    cv2.rectangle(img, p1, p2, color, grosor)

    if clOA in [0, 1]:
        etiquetaOA = "normal-dudoso"
    elif clOA in [2, 3]:
        etiquetaOA = "leve-moderado"
    else:
        etiquetaOA = "grave"

    cv2.putText(img, etiquetaOA, (p1[0] + 20, p1[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return img


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


# -------------------------------------------------
# 3) BODY REQUEST MODEL
# -------------------------------------------------

class ImageData(BaseModel):
    image: str


# -------------------------------------------------
# 4) ENDPOINT PRINCIPAL
# -------------------------------------------------

@app.post("/predict")
async def predict(payload: ImageData):
    print(">>> [API] /predict llamado")

    modeldetOP, modeldetOA = load_models()

    try:
        image_b64 = payload.image

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_original is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        certeza = 0

        clOP, prOP, crop, x1OP, y1OP, x2OP, y2OP = yolodetOPCrop(
            modeldetOP, img_original.copy(), certeza)

        if crop is None:
            raise HTTPException(status_code=400,
                                detail="No se detectó zona OP")

        clOA, prOA, x1OA, y1OA, x2OA, y2OA = yolodetOA(
            modeldetOA, crop, certeza)

        h, w = crop.shape[:2]
        crop_etiquetado = etiquetar(
            crop.copy(), clOP, 0, 0, w-1, h-1, clOA, x1OA, y1OA, x2OA, y2OA)

        return {
            "resultado": {
                "clase_op": "normal" if clOP == 0 else "osteoporosis",
                "prob_op": float(prOP),
                "clase_oa":
                    "normal-dudoso" if clOA in [0, 1]
                    else "leve-moderado" if clOA in [2, 3]
                    else "grave",
                "prob_oa": float(prOA),
            },
            "imagenOriginal": to_base64(img_original),
            "imagenProcesada": to_base64(crop),
            "imagenEtiquetada": to_base64(crop_etiquetado),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(">>> ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# 5) HEALTH ENDPOINT
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "message": "Backend FastAPI funcionando"}
