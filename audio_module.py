import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import json
from transformers import pipeline
from TTS.api import TTS

# Directory di cache per i modelli
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

class AudioProcessor:
    def __init__(self):
        # Inizializza Whisper
        self.whisper = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=-1,  # Usa CPU per questo modulo
            model_kwargs={"cache_dir": os.path.join(MODELS_CACHE_DIR, "whisper")},
            local_files_only=True,
        )
        
        # Inizializza TTS
        self.tts = TTS(
            model_name="tts_models/it/mai_female/glow-tts", 
            progress_bar=False,
            cache_dir=os.path.join(MODELS_CACHE_DIR, "tts")
        )
        
        # Audio queues
        self.audio_input_queue = queue.Queue()
        
        # Flag per il controllo dei thread
        self.running = False
    
    def start_audio_recording(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_recording_thread)
        self.audio_thread.start()
    
    def _audio_recording_thread(self):
        sample_rate = 16000
        channels = 1
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Error in audio recording: {status}")
            self.audio_input_queue.put(indata.copy())
        
        with sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate):
            while self.running:
                time.sleep(0.1)
    
    def stop_audio_recording(self):
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
    
    def process_audio_to_text(self):
        audio_chunks = []
        while not self.audio_input_queue.empty():
            audio_chunks.append(self.audio_input_queue.get())
        
        if not audio_chunks:
            return None
        
        audio_data = np.concatenate(audio_chunks)
        
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio_data, 16000)
        
        result = self.whisper(temp_file)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result["text"]
    
    def text_to_speech(self, text):
        temp_file = "temp_tts.wav"
        self.tts.tts_to_file(text=text, file_path=temp_file)
        
        data, samplerate = sf.read(temp_file)
        sd.play(data, samplerate)
        sd.wait()
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Server audio
if __name__ == "__main__":
    import sys
    
    print("Inizializzazione del processore audio...")
    audio_processor = AudioProcessor()
    print("Processore audio pronto.")
    
    # Funzione per registrare e trascrivere
    def record_and_transcribe():
        print("Avvio registrazione...")
        audio_processor.start_audio_recording()
        time.sleep(5)  # Registra per 5 secondi
        audio_processor.stop_audio_recording()
        
        text = audio_processor.process_audio_to_text()
        return text
    
    # Loop principale
    while True:
        command = input("Digita 'record' per registrare, 'speak' per parlare, o 'exit' per uscire: ")
        
        if command == "exit":
            break
        elif command == "record":
            text = record_and_transcribe()
            if text:
                print(f"Trascrizione: {text}")
                # Invia al modulo RAG
                print(json.dumps({"transcription": text}))
                sys.stdout.flush()
        elif command.startswith("speak:"):
            text = command[6:].strip()
            audio_processor.text_to_speech(text)
        else:
            print("Comando non riconosciuto")