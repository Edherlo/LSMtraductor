# lsm_model.py
import numpy as np
from keras.models import load_model

from preprocessing import normalize_keypoints
from loader import load_words_json
from config import MODEL_PATH, MODEL_FRAMES


class LSMModelService:
    """
    Servicio para cargar el modelo de LSM y hacer predicciones
    a partir de secuencias de keypoints.
    """

    def __init__(self):
        # Cargar modelo .keras
        self.model = load_model(MODEL_PATH)

        # Cargar etiquetas desde JSON
        self.words_text = load_words_json()

        # Frames que recibe el modelo
        self.num_frames = int(MODEL_FRAMES)

    def _prepare_sequence(self, kp_seq):
        """
        Normaliza la secuencia de keypoints a la longitud del modelo.
        """
        kp_normalized = normalize_keypoints(kp_seq, self.num_frames)
        kp_array = np.expand_dims(kp_normalized, axis=0)
        return kp_array

    def predict_from_keypoints_sequence(self, kp_seq, threshold=0.0):
        """
        kp_seq: lista de frames de keypoints capturados por Mediapipe.
        Retorna:
        {
            label: "hola",
            confidence: 0.94,
            index: 0,
            probs: [...]
        }
        """
        if not kp_seq:
            raise ValueError("La secuencia de keypoints está vacía.")

        input_array = self._prepare_sequence(kp_seq)

        # Predicción
        probs = self.model.predict(input_array)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        # Obtener la palabra según el JSON (keys ordenadas)
        keys = list(self.words_text.keys())
        label = self.words_text.get(keys[idx])

        # Filtro opcional
        if threshold > 0 and conf < threshold:
            return {
                "label": None,
                "confidence": conf,
                "index": idx,
                "probs": probs.tolist()
            }

        return {
            "label": label,
            "confidence": conf,
            "index": idx,
            "probs": probs.tolist()
        }
