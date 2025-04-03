# Esegui con: conda run -n audio_env python download_audio_models.py
import os
from transformers import pipeline
from TTS.api import TTS

# Directory di cache per i modelli
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

print("Scaricamento modello Whisper...")
whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    model_kwargs={"cache_dir": os.path.join(MODELS_CACHE_DIR, "whisper")}
)

print("Scaricamento modello TTS...")
tts = TTS(
    model_name="tts_models/it/mai_female/glow-tts", 
    progress_bar=True,
    cache_dir=os.path.join(MODELS_CACHE_DIR, "tts")
)

print("Download completato per i modelli audio!")