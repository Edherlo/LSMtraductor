# loader.py
import json
from config import WORDS_JSON_PATH

def load_words_json():
    """
    Devuelve un diccionario {id: palabra}
    usando el JSON proporcionado.
    """
    with open(WORDS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
