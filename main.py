import os
import cv2
import base64
import numpy as np
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ===========================
# APP
# ===========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# RUTAS DE MODELOS (backend/)
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

PATH_RECORTE = os.path.join(BACKEND_DIR, "recorte2.pt")
PATH_OP = os.path.join(BACKEND_DIR, "3clsOPfft.pt")
PATH_OA = os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt")

# ===========================
# CARGA DE MODELOS (Render-safe)
# ===========================
@lru_cache(maxsize=1)
def load_models():
    return (
        YOLO(PATH_RECORTE),
        YOLO(PATH_OP),
        YOLO(PATH_OA),
    )

# ===========================
# UTILIDADES BASE64
# ===========================
def decode_base64_image(b64: str):
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
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

# ===========================
# FUNCIONES ORIGINALES (ajustadas)
# ===========================
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

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = model(ms)
    cls = int(results[0].probs.top1)
    prob = float(results[0].probs.top1conf)
    return cls, prob

def yolodetOA(model, crop, certeza=0.0):
    results = model(crop)
    cls = []
    prob = []
    coords = []

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append(list(map(int, box.xyxy[0])))

    if not prob:
        return None

    i = int(np.argmax(prob))
    x1, y1, x2, y2 = coords[i]
    return cls[i], prob[i], x1, y1, x2, y2

def etiquetar2(img, clOP, c, clOA, oa_box):
    xOP1, yOP1, xOP2, yOP2 = c

    # OP
    cv2.rectangle(img, (xOP1, yOP1), (xOP2, yOP2), (255, 0, 0), 2)
    etiquetas_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"]
    cv2.putText(
        img,
        etiquetas_op[clOP],
        (xOP1, max(30, yOP1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # OA
    if oa_box:
        clOA, _, x1, y1, x2, y2 = oa_box
        cv2.rectangle(
            img,
            (xOP1 + x1, yOP1 + y1),
            (xOP1 + x2, yOP1 + y2),
            (0, 0, 255),
            2
        )
        etiquetas_oa = [
            "Sin Osteoartrosis",
            "OA dudoso",
            "OA leve",
            "OA moderado",
            "OA grave"
        ]
        cv2.putText(
            img,
            etiquetas_oa[clOA],
            (xOP1 + x1, max(30, yOP1 + y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    return img

# ===========================
# PIPELINE PRINCIPAL
# ===========================
def CorrerModelo(img):
    modelrecorte, modelOP, modelOA = load_models()

    img_original = img.copy()
    img_etiquetada = img.copy()
    imagen_procesada = None

    coor = yolorecorte(modelrecorte, img_original)
    if not coor:
        raise HTTPException(status_code=400, detail="No se detectó rodilla")

    clase_op = "normal"
    prob_op = 0.0
    clase_oa = "normal"
    prob_oa = 0.0

    for c in coor:
        x1, y1, x2, y2 = c

        # RECORTE LIMPIO (imagen procesada)
        crop = img_original[y1:y2, x1:x2].copy()
        imagen_procesada = crop.copy()

        clOP, probOP = yolodetOPCrop(modelOP, crop)
        clase_op = ["normal", "osteopenia", "osteoporosis"][clOP]
        prob_op = probOP

        oa = yolodetOA(modelOA, crop)

        if oa:
            clOA, probOA, *_ = oa
            clase_oa = ["normal", "dudoso", "leve", "moderado", "grave"][clOA]
            prob_oa = probOA
        else:
            clOA = None

        img_etiquetada = etiquetar2(img_etiquetada, clOP, c, clOA, oa)

    return imagen_procesada, img_etiquetada, clase_op, prob_op, clase_oa, prob_oa

# ===========================
# API (MISMO CONTRATO QUE TU FRONTEND)
# ===========================
@app.post("/predict")
async def predict(req: Request):
    data = await req.json()

    if "image" not in data:
        raise HTTPException(status_code=400, detail="No se recibió imagen")

    img = decode_base64_image(data["image"])

    img_proc, img_etq, clase_op, prob_op, clase_oa, prob_oa = CorrerModelo(img)

    return {
        "resultado": {
            "clase_op": clase_op,
            "prob_op": prob_op,
            "clase_oa": clase_oa,
            "prob_oa": prob_oa,
        },
        "imagenProcesada": to_base64(img_proc),
        "imagenEtiquetada": to_base64(img_etq),
    }

@app.get("/")
def health():
    return {"status": "ok"}
