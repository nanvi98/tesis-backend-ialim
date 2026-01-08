import os
import cv2
import numpy as np
import base64
import gc

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


base_path = os.path.dirname(__file__)

path_detOP = os.path.join(base_path, "backend", "det 2cls R2 0.pt")
path_detOA = os.path.join(base_path, "backend", "OAyoloIR4AH.pt")


def load_model_op():
    return YOLO(path_detOP)


def load_model_oa():
    return YOLO(path_detOA)


class ImageData(BaseModel):
    image: str


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

    i = prob.index(max(prob))
    return cls[i], prob[i], crops[i], *coords[i]


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

    i = prob.index(max(prob))
    return cls[i], prob[i], *coords[i]


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


@app.post("/predict")
async def predict(payload: ImageData):

    try:
        image_b64 = payload.image

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(400, "Imagen corrupta")

        certeza = 0

        # -------- LOAD ONLY OP MODEL --------
        modelOP = load_model_op()
        clOP, prOP, crop, x1, y1, x2, y2 = yolodetOPCrop(modelOP, img, certeza)

        del modelOP
        gc.collect()

        if crop is None:
            raise HTTPException(400, "No se detectó zona OP")

        # -------- LOAD ONLY OA MODEL --------
        modelOA = load_model_oa()
        clOA, prOA, xa1, ya1, xa2, ya2 = yolodetOA(modelOA, crop, certeza)

        del modelOA
        gc.collect()

        h, w = crop.shape[:2]

        return {
            "resultado": {
                "clase_op": "normal" if clOP == 0 else "osteoporosis",
                "prob_op": float(prOP),
                "clase_oa": (
                    "normal-dudoso" if clOA in [0, 1]
                    else "leve-moderado" if clOA in [2, 3]
                    else "grave"
                ),
                "prob_oa": float(prOA),
            },
            "imagenOriginal": to_base64(img),
            "imagenProcesada": to_base64(crop),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/")
def health():
    return {"status": "ok"}
