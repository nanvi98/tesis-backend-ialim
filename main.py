import os
import cv2
import numpy as np
import base64
from functools import lru_cache

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


BASE = os.path.dirname(__file__)
PATH_OP = os.path.join(BASE, "backend", "det 2cls R2 0.pt")
PATH_OA = os.path.join(BASE, "backend", "OAyoloIR4AH.pt")


@lru_cache(maxsize=1)
def load_models():
    model_op = YOLO(PATH_OP)
    model_oa = YOLO(PATH_OA)
    return model_op, model_oa


class ImageData(BaseModel):
    image: str


def decode_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]

    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen corrupta")

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen corrupta")

    return img


@app.post("/predict")
async def predict(payload: ImageData):
    img = decode_image(payload.image)

    model_op, model_oa = load_models()

    op = model_op(img)[0]
    if len(op.boxes) == 0:
        raise HTTPException(status_code=400, detail="No se detectó OP")

    box = op.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]

    oa = model_oa(crop)[0]

    return {
        "ok": True,
    }


@app.get("/")
def health():
    return {"ok": True}
