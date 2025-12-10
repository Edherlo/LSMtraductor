# main.py - VERSIÓN REAL CORREGIDA PARA 63 KEYPOINTS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List
import time
import logging
import numpy as np
from keras.models import load_model
import json

from preprocessing import normalize_keypoints
from LSMtraductor.constants import MODEL_PATH, WORDS_JSON_PATH, MODEL_FRAMES, LENGTH_KEYPOINTS

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= INICIALIZAR APP =============
app = FastAPI(
    title="LSM API - Lengua de Señas Mexicana",
    description="API para traducir señas a texto usando keypoints de MANOS",
    version="2.0.0"
)

# CORS para Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= CARGAR MODELO =============
logger.info(f"Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH)
logger.info("✅ Modelo cargado exitosamente")

# Cargar palabras
with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
    word_data = json.load(f)
    word_ids = word_data.get('word_ids', [])
logger.info(f"✅ Cargadas {len(word_ids)} palabras: {word_ids}")

# ============= MODELOS PYDANTIC =============
class KeypointSequence(BaseModel):
    sequence: List[List[float]]
    
    @validator('sequence')
    def validate_sequence_structure(cls, v):
        if not v:
            raise ValueError("La secuencia no puede estar vacía")
        
        if len(v) < 5:
            raise ValueError(
                f"Se requieren al menos 5 frames, recibidos: {len(v)}"
            )
        
        # Validar longitud consistente
        first_length = len(v[0])
        for i, frame in enumerate(v):
            if len(frame) != first_length:
                raise ValueError(
                    f"Inconsistencia en frame {i}: "
                    f"esperados {first_length} keypoints, recibidos {len(frame)}"
                )
        
        return v

# ============= FUNCIONES AUXILIARES =============
def validate_keypoints_values(sequence: List[List[float]]) -> tuple:
    """Valida formato y contenido de keypoints"""
    errors = []
    warnings = []
    
    # Verificar longitud esperada
    actual_length = len(sequence[0]) if sequence else 0
    
    if actual_length != LENGTH_KEYPOINTS:
        errors.append(
            f"❌ Longitud incorrecta: esperados {LENGTH_KEYPOINTS} keypoints "
            f"(21 landmarks × 3 valores), recibidos {actual_length}"
        )
        return False, errors, warnings
    
    # Estadísticas
    sequence_array = np.array(sequence)
    empty_frames = 0
    
    for i, frame in enumerate(sequence):
        frame_arr = np.array(frame)
        
        # Contar frames vacíos
        if np.count_nonzero(frame_arr) == 0:
            empty_frames += 1
        
        # Verificar rango válido [0, 1] para coordenadas normalizadas
        if np.any(frame_arr < -0.1) or np.any(frame_arr > 1.1):
            warnings.append(f"Frame {i} con valores fuera de rango [0, 1]")
    
    # Validar frames vacíos
    total_frames = len(sequence)
    if empty_frames == total_frames:
        errors.append("❌ Todos los frames están vacíos (sin detección de manos)")
    elif empty_frames > total_frames * 0.3:
        warnings.append(
            f"⚠️ {empty_frames}/{total_frames} frames vacíos "
            f"({empty_frames/total_frames*100:.1f}%)"
        )
    
    # Log warnings
    for warning in warnings:
        logger.warning(warning)
    
    return len(errors) == 0, errors, warnings

def predict_from_sequence(kp_seq: List[List[float]], threshold: float = 0.0):
    """Realiza predicción a partir de secuencia de keypoints"""
    
    # Normalizar a MODEL_FRAMES frames
    kp_normalized = normalize_keypoints(kp_seq, MODEL_FRAMES)
    kp_array = np.expand_dims(kp_normalized, axis=0)
    
    # Predicción
    probs = model.predict(kp_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    
    # Obtener palabra
    if idx < len(word_ids):
        word_id = word_ids[idx]
    else:
        word_id = "unknown"
    
    # Aplicar threshold
    if threshold > 0 and conf < threshold:
        return {
            "label": None,
            "confidence": conf,
            "index": idx,
            "below_threshold": True
        }
    
    return {
        "label": word_id,
        "confidence": conf,
        "index": idx,
        "below_threshold": False
    }

# ============= ENDPOINTS =============
@app.get("/")
async def root():
    return {
        "message": "LSM Translation API - Hand Keypoints Only",
        "version": "2.0.0",
        "status": "online",
        "model": {
            "keypoints_type": "hands_only",
            "keypoints_per_frame": LENGTH_KEYPOINTS,
            "frames_required": MODEL_FRAMES
        }
    }

@app.get("/health")
async def health_check():
    """Health check del servicio"""
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "config": {
                "keypoints_per_frame": LENGTH_KEYPOINTS,
                "frames_required": MODEL_FRAMES,
                "total_classes": len(word_ids),
                "classes": word_ids
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )

@app.get("/info")
async def model_info():
    """Información detallada del modelo"""
    return {
        "model": {
            "type": "LSTM",
            "input_shape": f"({MODEL_FRAMES}, {LENGTH_KEYPOINTS})",
            "output_classes": len(word_ids),
            "classes": word_ids
        },
        "keypoints": {
            "type": "MediaPipe Holistic - HANDS ONLY",
            "format": "right_hand OR left_hand",
            "landmarks_per_hand": 21,
            "values_per_landmark": 3,  # x, y, z
            "total_per_frame": LENGTH_KEYPOINTS
        },
        "preprocessing": {
            "normalization": f"interpolate/downsample to {MODEL_FRAMES} frames",
            "method": "linear interpolation or frame skipping"
        }
    }

@app.post("/predict")
async def predict(sequence_data: KeypointSequence):
    """
    Predice la seña a partir de keypoints de MANOS
    
    **Formato esperado:**
    - Cada frame: 63 valores (21 landmarks × 3 coordenadas)
    - Mínimo 5 frames, máximo ilimitado (se normalizará a 15)
    - Valores en rango [0, 1] (normalizados por MediaPipe)
    
    **Ejemplo:**
    ```json
    {
        "sequence": [
            [0.5, 0.3, 0.1, ...],  // Frame 1: 63 valores
            [0.5, 0.3, 0.1, ...],  // Frame 2: 63 valores
            ...
        ]
    }
    ```
    """
    start_time = time.time()
    
    try:
        frames_received = len(sequence_data.sequence)
        keypoints_per_frame = len(sequence_data.sequence[0])
        
        logger.info(
            f"📥 Petición recibida: {frames_received} frames × "
            f"{keypoints_per_frame} keypoints"
        )
        
        # Validar valores
        is_valid, errors, warnings = validate_keypoints_values(
            sequence_data.sequence
        )
        
        if not is_valid:
            logger.error(f"❌ Validación fallida: {errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Datos inválidos",
                    "details": errors,
                    "received": {
                        "frames": frames_received,
                        "keypoints_per_frame": keypoints_per_frame
                    },
                    "expected": {
                        "keypoints_per_frame": LENGTH_KEYPOINTS,
                        "min_frames": 5
                    }
                }
            )
        
        # Predicción
        pred = predict_from_sequence(sequence_data.sequence, threshold=0.0)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Respuesta
        response = {
            "label": pred["label"],
            "confidence": pred["confidence"],
            "index": pred["index"],
            "metadata": {
                "frames_received": frames_received,
                "frames_used": MODEL_FRAMES,
                "keypoints_per_frame": keypoints_per_frame,
                "inference_time_ms": round(inference_time, 2),
                "warnings": warnings,
                "timestamp": time.time()
            }
        }
        
        logger.info(
            f"✅ Predicción: '{response['label']}' "
            f"({response['confidence']:.2%}) en {inference_time:.0f}ms"
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error interno",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/validate")
async def validate_only(sequence_data: KeypointSequence):
    """
    Valida formato sin hacer predicción (útil para debugging)
    """
    frames_received = len(sequence_data.sequence)
    keypoints_per_frame = len(sequence_data.sequence[0])
    
    is_valid, errors, warnings = validate_keypoints_values(
        sequence_data.sequence
    )
    
    sequence_array = np.array(sequence_data.sequence)
    
    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "received": {
            "frames": frames_received,
            "keypoints_per_frame": keypoints_per_frame
        },
        "expected": {
            "keypoints_per_frame": LENGTH_KEYPOINTS,
            "frames_for_model": MODEL_FRAMES,
            "keypoint_format": "21 hand landmarks × 3 values (x, y, z)"
        },
        "statistics": {
            "mean": float(sequence_array.mean()),
            "std": float(sequence_array.std()),
            "min": float(sequence_array.min()),
            "max": float(sequence_array.max()),
            "zeros_percentage": float(
                (sequence_array == 0).sum() / sequence_array.size * 100
            )
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Iniciando LSM API Server...")
    logger.info(f"📊 Modelo: {MODEL_PATH}")
    logger.info(f"🤚 Formato: {LENGTH_KEYPOINTS} keypoints (MANOS SOLAMENTE)")
    logger.info(f"🎬 Frames requeridos: {MODEL_FRAMES}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )