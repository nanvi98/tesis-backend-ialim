import os
import cv2
import base64
import numpy as np
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
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
# RUTAS DE MODELOS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

PATH_RECORTE = os.path.join(BACKEND_DIR, "recorte2.pt")
PATH_OP = os.path.join(BACKEND_DIR, "3clsOPfft.pt")
PATH_OA = os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt")

# ---------------------------
# CARGA DE MODELOS
# ---------------------------
@lru_cache(maxsize=1)
def load_models():
    return (
        YOLO(PATH_RECORTE),
        YOLO(PATH_OP),
        YOLO(PATH_OA),
    )

# ---------------------------
# UTILIDADES
# ---------------------------
def decode_base64_image(b64: str):
    if "," in b64:
        b64 = b64.split(",")[1]

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise ValueError("Base64 inválido")

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Imagen inválida")

    return img

def to_base64(img):
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("No se pudo codificar imagen")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ---------------------------
# MODELOS
# ---------------------------
def yolorecorte(model, img):
    results = model(img)
    coor = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor

def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fftshift(np.fft.fft2(crop))
    ms = (20 * np.log(np.abs(f) + 1)).astype(np.uint8)

    results = model(ms)
    cls = int(results[0].probs.top1)
    prob = float(results[0].probs.top1conf)
    return cls, prob

def yolodetOA(model, crop):
    results = model(crop)
    best = None
    best_conf = -1.0

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > best_conf:
                best = box
                best_conf = conf

    if best is None:
        return None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    return int(best.cls[0]), float(best.conf[0]), x1, y1, x2, y2

def etiquetar(img, clOP, c, clOA, oa_box):
    x1, y1, x2, y2 = c
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    cv2.putText(img, etiquetas_op[clOP], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if oa_box:
        cl, _, ox1, oy1, ox2, oy2 = oa_box
        cv2.rectangle(
            img,
            (x1 + ox1, y1 + oy1),
            (x1 + ox2, y1 + oy2),
            (0, 0, 255),
            2
        )

    return img

def CorrerModelo(img):
    modelrecorte, modelOP, modelOA = load_models()

    coor = yolorecorte(modelrecorte, img)
    if not coor:
        raise HTTPException(status_code=400, detail="No se detectó rodilla")

    clase_op = "normal"
    prob_op = 0.0
    clase_oa = "normal"
    prob_oa = 0.0

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        clOP, probOP = yolodetOPCrop(modelOP, crop)
        oa = yolodetOA(modelOA, crop)

        clase_op = ["normal", "osteopenia", "osteoporosis"][clOP]
        prob_op = probOP

        if oa:
            clOA, probOA, *_ = oa
            clase_oa = ["normal", "dudoso", "leve", "moderado", "grave"][clOA]
            prob_oa = probOA
            img = etiquetar(img, clOP, c, clOA, oa)

    return img, clase_op, prob_op, clase_oa, prob_oa

# ---------------------------
# API (IGUAL AL BACKEND VIEJO)
# ---------------------------
@app.post("/predict")
async def predict(req: Request):
    data = await req.json()

    if "image" not in data:
        raise HTTPException(status_code=400, detail="No se recibió imagen")

    try:
        img = decode_base64_image(data["image"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_res, clase_op, prob_op, clase_oa, prob_oa = CorrerModelo(img)

    return {
        "resultado": {
            "clase_op": clase_op,
            "prob_op": prob_op,
            "clase_oa": clase_oa,
            "prob_oa": prob_oa,
        },
        "imagenProcesada": to_base64(img),
        "imagenEtiquetada": to_base64(img_res),
    }

@app.get("/")
def health():
    return {"status": "ok"}
