from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from ultralytics import YOLO
import io


# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------

modelrecorte = YOLO('backend/recorte2.pt')
modeldetOP = YOLO('backend/3clsOPfft.pt')
modeldetOA = YOLO('backend/OAyoloR4cls5.pt')


# ---------------------------
# 2) FUNCIONES ORIGINALES
# ---------------------------

def yolorecorte(modelrecorte, img):
    results = modelrecorte(img)
    coor = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])

    print("Rodillas detectadas:", coor)
    return coor


def yolodetOPCrop(modeldetOPfft, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = modeldetOPfft(ms)

    for result in results:
        cls = int(result.probs.top1)
        prob = float(result.probs.top1conf)
        return cls, prob

    return -1, 0.0


def yolodetOA(modeldet, crop, certeza):
    results = modeldet(crop)

    cls = []
    prob = []
    coords = []

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coords.append((x1, y1, x2, y2))

    # 🔴 PROTECCIÓN CLAVE
    if len(prob) == 0:
        return -1, 0.0, 0, 0, 0, 0

    idx = prob.index(max(prob))
    x1, y1, x2, y2 = coords[idx]

    return cls[idx], prob[idx], x1, y1, x2, y2


def etiquetar2(imagen, clOP, xOP1, yOP1, xOP2, yOP2,
               clOA, xOA1, yOA1, xOA2, yOA2):

    color = (255, 0, 0)
    grosor = 2

    # ---- Osteoporosis ----
    cv2.rectangle(imagen, (xOP1, yOP1), (xOP2, yOP2), color, grosor)

    if clOP == 0:
        etiqueta = 'Sin osteoporosis'
    elif clOP == 1:
        etiqueta = 'Osteopenia'
    elif clOP == 2:
        etiqueta = 'Osteoporosis'
    else:
        etiqueta = 'OP no concluyente'

    cv2.putText(
        imagen, etiqueta, (xOP1, yOP1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    # ---- Osteoartritis (solo si hubo detección) ----
    if clOA != -1:
        p1 = (xOP1 + xOA1, yOP1 + yOA1)
        p2 = (xOP1 + xOA2, yOP1 + yOA2)
        cv2.rectangle(imagen, p1, p2, color, grosor)

        if clOA == 0:
            etiqueta = 'Sin OA'
        elif clOA == 1:
            etiqueta = 'OA dudoso'
        elif clOA == 2:
            etiqueta = 'OA leve'
        elif clOA == 3:
            etiqueta = 'OA moderado'
        else:
            etiqueta = 'OA grave'

        cv2.putText(
            imagen, etiqueta, (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    return imagen


def CorrerModelo(img):
    certeza = 0
    coor = yolorecorte(modelrecorte, img)

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2], :]

        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        clOA, _, xOA1, yOA1, xOA2, yOA2 = yolodetOA(modeldetOA, crop, certeza)

        img = etiquetar2(
            img,
            clOP, c[0], c[1], c[2], c[3],
            clOA, xOA1, yOA1, xOA2, yOA2
        )

    return img


# ---------------------------
# 3) API FASTAPI
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(request: Request):
    form = await request.form()

    upload = None
    for v in form.values():
        if hasattr(v, "filename"):
            upload = v
            break

    if upload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No se recibió ningún archivo"}
        )

    contents = await upload.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Imagen inválida"}
        )

    result_img = CorrerModelo(img)

    _, img_encoded = cv2.imencode(".jpg", result_img)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )
