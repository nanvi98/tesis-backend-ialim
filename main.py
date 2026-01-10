from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# -------------------------
# MODELOS YOLO
# -------------------------
# Ajusta los nombres si tus .pt se llaman distinto
model_recorte = YOLO("backend/corte0.pt")
model_op = YOLO("backend/det 2cls R2 0.pt")
model_oa = YOLO("backend/OAyoloIR4AH.pt")

# -------------------------
# SCHEMA
# -------------------------
class ImageRequest(BaseModel):
    image: str  # base64 (con o sin data:image/...)


# -------------------------
# UTILS
# -------------------------
def decode_base64_image(b64: str):
    try:
        if "," in b64:
            b64 = b64.split(",")[1]

        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        return img

    except Exception:
        raise HTTPException(status_code=400, detail="Imagen corrupta o inválida")


# -------------------------
# ENDPOINTS
# -------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "Backend YOLO activo"}


@app.post("/predict")
def predict(req: ImageRequest):
    img = decode_base64_image(req.image)

    # ---------- 1) RECORTE ----------
    recorte = model_recorte(img)[0]

    if len(recorte.boxes) == 0:
        return {
            "osteoporosis": False,
            "osteoartritis": False,
            "message": "No se detectó región de interés"
        }

    # Tomar el primer bbox
    x1, y1, x2, y2 = map(int, recorte.boxes[0].xyxy[0])
    roi = img[y1:y2, x1:x2]

    # ---------- 2) OSTEOPOROSIS ----------
    res_op = model_op(roi)[0]
    has_op = any(box.conf.item() > 0.5 for box in res_op.boxes)

    # ---------- 3) OSTEOARTRITIS ----------
    res_oa = model_oa(roi)[0]
    has_oa = any(box.conf.item() > 0.5 for box in res_oa.boxes)

    return {
        "osteoporosis": has_op,
        "osteoartritis": has_oa
    }
