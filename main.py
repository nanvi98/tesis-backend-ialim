from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend Firebase / local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MODELOS YOLO
# -------------------------------------------------
# ⚠️ Rutas relativas al repo
model_recorte = YOLO("backend/corte0.pt")
model_op = YOLO("backend/det 2cls R2 0.pt")
model_oa = YOLO("backend/OAyoloIR4AH.pt")

# -------------------------------------------------
# REQUEST
# -------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # base64 data:image/...

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def decode_base64_image(data: str) -> np.ndarray:
    try:
        if "," in data:
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen corrupta o inválida")


def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"

# -------------------------------------------------
# ENDPOINT
# -------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):

    # 1️⃣ Decodificar imagen original
    img = decode_base64_image(req.image)

    # 2️⃣ RECORTE DE RODILLA
    recorte_result = model_recorte(img)[0]

    if len(recorte_result.boxes) > 0:
        box = recorte_result.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        img_crop = img[y1:y2, x1:x2]
    else:
        img_crop = img.copy()

    imagen_procesada = encode_image(img_crop)

    # 3️⃣ MODELOS OP y OA SOBRE RECORTE
    res_op = model_op(img_crop)[0]
    res_oa = model_oa(img_crop)[0]

    clase_op = "normal"
    prob_op = 0.0
    if len(res_op.boxes) > 0:
        clase_op = "osteoporosis"
        prob_op = float(res_op.boxes[0].conf[0])

    clase_oa = "normal"
    prob_oa = 0.0
    if len(res_oa.boxes) > 0:
        clase_oa = "osteoartritis"
        prob_oa = float(res_oa.boxes[0].conf[0])

    # 4️⃣ IMAGEN ETIQUETADA (CAJAS YOLO)
    img_etiquetada = img_crop.copy()

    if len(res_op.boxes) > 0:
        img_etiquetada = res_op.plot(img=img_etiquetada)

    if len(res_oa.boxes) > 0:
        img_etiquetada = res_oa.plot(img=img_etiquetada)

    imagen_etiquetada = encode_image(img_etiquetada)

    # 5️⃣ RESPUESTA
    return {
        "resultado": {
            "clase_op": clase_op,
            "prob_op": round(prob_op, 3),
            "clase_oa": clase_oa,
            "prob_oa": round(prob_oa, 3),
        },
        "imagenProcesada": imagen_procesada,
        "imagenEtiquetada": imagen_etiquetada,
    }

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "Backend YOLO activo"}
