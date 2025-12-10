import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 63
MODEL_FRAMES = 15

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

words_text = {
    "hola": "HOLA",
    "adios": "ADIÓS",
    "de_nada": "DE NADA",
    "buen_dia": "BUEN DÍA",
    "gracias": "GRACIAS",
    "escuche": "ESCUCHÉ",
    "olvide": "OLVIDÉ",
    "nos_vemos": "NOS VEMOS",
    "te_quiero": "TE QUIERO",
    "como": "CÓMO",
    "estas": "ESTÁS",
    "yo": "YO",
    "tu": "TÚ",
    "mio": "MÍO",
    "tener": "TENER",
    "pequeño": "PEQUEÑO",
    "perdon": "PERDÓN",
    "voy_a_mi": "VOY A MI",
    "casa": "CASA",
    "cigarro": "CIGARRO",
    "cuanto": "CUÁNTO",
    "cuesta": "CUESTA",
}