from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

app = FastAPI()

# -------- MODELOS (AJUSTA RUTAS SI ES NECESARIO) ----------
model_crop = YOLO("backend/corte0.pt")
model_op = YOLO("backend/det 2cls R2 0.pt")
model_oa = YOLO("backend/OAyoloIR4AH.pt")


# -------- REQUEST BODY ----------
class ImageRequest(BaseModel):
    image: str


# -------- DECODIFICAR BASE64 SIN ERRORES ----------
def decode_image(image_string):

    # Si viene con "data:image/png;base64,"
    if "," in image_string:
        image_string = image_string.split(",")[1]

    # Corregir padding faltante
    missing_padding = len(image_string) % 4
    if missing_padding:
        image_string += "=" * (4 - missing_padding)

    try:
        img_bytes = base64.b64decode(image_string)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return np.array(image)

    except Exception:
        raise HTTPException(status_code=400, detail="Imagen corrupta")


# -------- ENDPOINT PRINCIPAL ----------
@app.post("/predict")
async def predict(data: ImageRequest):

    img = decode_image(data.image)

    # Convertir a BGR para OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---------- TU PROCESO EJEMPLO ----------
    # Recorte
    crop_results = model_crop(img_bgr)

    # Osteoporosis
    op_results = model_op(img_bgr)

    # Osteoartritis
    oa_results = model_oa(img_bgr)

    # Aquí regresas lo que tú necesites.
    return {
        "status": "ok",
        "crop_detections": len(crop_results[0].boxes),
        "op_detections": len(op_results[0].boxes),
        "oa_detections": len(oa_results[0].boxes)
    }


@app.get("/")
def home():
    return {"message": "API funcionando 👍"}
