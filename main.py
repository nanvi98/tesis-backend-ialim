from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64

# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------

modelrecorte = YOLO("backend/recorte2.pt")
modeldetOP = YOLO("backend/3clsOPfft.pt")
modeldetOA = YOLO("backend/OAyoloR4cls5.pt")

# ---------------------------
# 2) FUNCIONES
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

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = model(ms)
    r = results[0]
    return int(r.probs.top1), float(r.probs.top1conf)

def yolodetOA(model, crop):
    results = model(crop)
    best = max(results[0].boxes, key=lambda b: b.conf[0])
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    return int(best.cls), float(best.conf[0]), x1, y1, x2, y2

def etiquetar(img, c, op, oa):
    x1, y1, x2, y2 = c
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

    op_text = ["Sin osteoporosis","Osteopenia","Osteoporosis"][op[0]]
    cv2.putText(img, op_text, (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    ox1, oy1, ox2, oy2 = oa[2:]
    cv2.rectangle(img, (x1+ox1,y1+oy1), (x1+ox2,y1+oy2), (0,0,255), 2)

    oa_text = ["Sin OA","OA dudoso","OA leve","OA moderado","OA grave"][oa[0]]
    cv2.putText(img, oa_text, (x1+ox1,y1+oy1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

def correr_modelo(img):
    coor = yolorecorte(modelrecorte, img)
    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        op = yolodetOPCrop(modeldetOP, crop)
        oa = yolodetOA(modeldetOA, crop)
        etiquetar(img, c, op, oa)
    return img

# ---------------------------
# 3) API
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    result = correr_modelo(img)

    _, buffer = cv2.imencode(".jpg", result)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "imagen_procesada": f"data:image/jpeg;base64,{img_base64}",
        "imagen_etiquetada": f"data:image/jpeg;base64,{img_base64}"
    })
