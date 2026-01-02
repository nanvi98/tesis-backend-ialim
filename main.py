# import os
# import cv2
# import numpy as np
# import base64
# from firebase_functions import https_fn, options
# from firebase_admin import initialize_app
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # luego lo cerramos si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # ---------------------------
# # 0) Configuración Inicial de Firebase
# # ---------------------------
# initialize_app()

# # Usamos Flask en lugar de FastAPI para compatibilidad nativa
# app = Flask(__name__)
# CORS(app,resources={r"/*": {"origins": "*"}}) # Habilitar CORS para que el frontend pueda conectarse

# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     return response

# ---------------------------
# 1) Cargar modelos
# ---------------------------
base_path = os.path.dirname(__file__)

# Rutas absolutas para los modelos
path_corte = os.path.join(base_path, 'backend', 'corte0.pt')
path_detOP = os.path.join(base_path, 'backend', 'det 2cls R2 0.pt')
path_detOA = os.path.join(base_path, 'backend', 'OAyoloIR4AH.pt')

print(f"--- Cargando modelos desde: {base_path} ---")

try:
    # Cargamos los modelos al iniciar la instancia
    # modelrecorte = YOLO(path_corte) # Descomenta si usas el recorte
    modeldetOP = YOLO(path_detOP)
    modeldetOA = YOLO(path_detOA)
except Exception as e:
    print(f"Error FATAL cargando modelos: {e}")

# ---------------------------
# 2) Funciones de procesamiento (Tu lógica intacta)
# ---------------------------

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
    
    if not prob: return 0, 0, None, 0, 0, 0, 0 # Retornamos None en crop para validar después
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
    
    if not prob: return 0, 0, 0, 0, 0, 0
    x = prob.index(max(prob))
    return cls[x], prob[x], *coords[x]

def etiquetar(img, clOP, xOP1, yOP1, xOP2, yOP2, clOA, xOA1, yOA1, xOA2, yOA2):
    color = (255, 0, 0)
    grosor = 2
    # OP
    cv2.rectangle(img, (xOP1, yOP1), (xOP2, yOP2), color, grosor)
    etiqueta = 'normal' if clOP == 0 else 'osteoporosis'
    cv2.putText(img, etiqueta, (xOP1 + 20, yOP1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # OA
    p1 = (xOP1 + xOA1, yOP1 + yOA1)
    p2 = (xOP1 + xOA2, yOP1 + yOA2)
    cv2.rectangle(img, p1, p2, color, grosor)
    
    if clOA in [0, 1]: etiquetaOA = "normal-dudoso"
    elif clOA in [2, 3]: etiquetaOA = "leve-moderado"
    else: etiquetaOA = "grave"

    cv2.putText(img, etiquetaOA, (p1[0] + 20, p1[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

def to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ---------------------------
# 3) API (Adaptada a Flask + JSON)
# ---------------------------

# ---------------------------
# 3) API (Adaptada a Flask + JSON)
# ---------------------------

# @app.route('/backend/predict', methods=['POST', 'OPTIONS'])
# def predict():
#     print(">>> [API] Llamada a /backend/predict")

#     # ⭐ Preflight CORS
#     if request.method == "OPTIONS":
#         return jsonify({"status": "OK preflight"}), 200

#     try:
#         # 1. Leer JSON
#         data_json = request.get_json()
#         print(">>> [API] JSON recibido:", "OK" if data_json else "VACÍO")

#         if not data_json or 'image' not in data_json:
#             print(">>> [API] ERROR: no se recibió 'image'")
#             return jsonify({"error": "No se recibió la imagen"}), 400

#         # 2. Decodificar Base64
#         image_b64 = data_json['image']
#         if "," in image_b64:
#             image_b64 = image_b64.split(",")[1]

#         image_bytes = base64.b64decode(image_b64)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if img_original is None:
#             print(">>> [API] ERROR: imagen corrupta")
#             return jsonify({"error": "Imagen corrupta o ilegible"}), 400

#         certeza = 0

#         # ---- YOLO OP ----
#         print(">>> [API] Ejecutando YOLO OP...")
#         clOP, prOP, crop, x1OP, y1OP, x2OP, y2OP = yolodetOPCrop(
#             modeldetOP, img_original.copy(), certeza
#         )

#         if crop is None:
#             return jsonify({"error": "No se detectó zona de interés (OP)"}), 400

#         # ---- YOLO OA ----
#         print(">>> [API] Ejecutando YOLO OA...")
#         clOA, prOA, x1OA, y1OA, x2OA, y2OA = yolodetOA(
#             modeldetOA, crop, certeza
#         )

#         # ---- Etiquetado ----
#         h, w = crop.shape[:2]
#         crop_etiquetado = etiquetar(
#             crop.copy(), clOP, 0, 0, w-1, h-1, clOA, x1OA, y1OA, x2OA, y2OA
#         )

#         print(">>> [API] Respuesta OK lista para enviar")

#         return jsonify({
#             "resultado": {
#                 "clase_op": "normal" if clOP == 0 else "osteoporosis",
#                 "prob_op": float(prOP),
#                 "clase_oa":
#                     "normal-dudoso" if clOA in [0, 1]
#                     else "leve-moderado" if clOA in [2, 3]
#                     else "grave",
#                 "prob_oa": float(prOA)
#             },
#             "imagenOriginal": to_base64(img_original),
#             "imagenProcesada": to_base64(crop),
#             "imagenEtiquetada": to_base64(crop_etiquetado)
#         }), 200
@app.post("/predict")
async def predict(request: Request):
    print(">>> [API] Llamada a /predict")

    # 1. Leer JSON
    data_json = await request.json()
    print(">>> [API] JSON recibido:", "OK" if data_json else "VACÍO")

    if not data_json or 'image' not in data_json:
        print(">>> [API] ERROR: no se recibió 'image'")
        return {"error": "No se recibió la imagen"}

    # 2. Decodificar Base64
    image_b64 = data_json['image']
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_original is None:
        print(">>> [API] ERROR: imagen corrupta")
        return {"error": "Imagen corrupta o ilegible"}

    certeza = 0

    # ---- YOLO OP ----
    print(">>> [API] Ejecutando YOLO OP...")
    clOP, prOP, crop, x1OP, y1OP, x2OP, y2OP = yolodetOPCrop(
        modeldetOP, img_original.copy(), certeza
    )

    if crop is None:
        return {"error": "No se detectó zona de interés (OP)"}

    # ---- YOLO OA ----
    print(">>> [API] Ejecutando YOLO OA...")
    clOA, prOA, x1OA, y1OA, x2OA, y2OA = yolodetOA(
        modeldetOA, crop, certeza
    )

    # ---- Etiquetado ----
    h, w = crop.shape[:2]
    crop_etiquetado = etiquetar(
        crop.copy(), clOP, 0, 0, w-1, h-1, clOA, x1OA, y1OA, x2OA, y2OA
    )

    print(">>> [API] Respuesta OK lista para enviar")

    return {
        "resultado": {
            "clase_op": "normal" if clOP == 0 else "osteoporosis",
            "prob_op": float(prOP),
            "clase_oa":
                "normal-dudoso" if clOA in [0, 1]
                else "leve-moderado" if clOA in [2, 3]
                else "grave",
            "prob_oa": float(prOA)
        },
        "imagenOriginal": to_base64(img_original),
        "imagenProcesada": to_base64(crop),
        "imagenEtiquetada": to_base64(crop_etiquetado)
    }

    # except Exception as e:
    #     print(">>> ERROR EN PREDICCIÓN:", e)
    #     return jsonify({"error": str(e)}), 500

@app.get("/")
def health():
    return {"status": "ok", "message": "Backend FastAPI funcionando"}

# # ---------------------------
# # 4) Configuración para Firebase Functions
# # ---------------------------
# # Esto funciona porque 'app' es de Flask. Con FastAPI esto fallaba.
# @https_fn.on_request(
#     memory=options.MemoryOption.GB_2,
#     timeout_sec=300,
#     region="us-central1"
# )
# def api(req: https_fn.Request) -> https_fn.Response:
#     with app.request_context(req.environ):
#         return app.full_dispatch_request()