import os
import json
import subprocess
import threading
import queue
import time

# Code per la comunicazione tra processi
rag_to_audio_queue = queue.Queue()
audio_to_rag_queue = queue.Queue()

def start_rag_process():
    """Avvia il processo RAG in un ambiente conda."""
    process = subprocess.Popen(
        ["conda", "run", "-n", "rag_env", "python", "rag_module.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Thread per leggere l'output
    def read_output():
        for line in process.stdout:
            try:
                data = json.loads(line)
                if "response" in data:
                    rag_to_audio_queue.put(data["response"])
            except json.JSONDecodeError:
                print(f"RAG output (non-JSON): {line.strip()}")
    
    threading.Thread(target=read_output, daemon=True).start()
    
    # Thread per leggere stderr
    def read_stderr():
        for line in process.stderr:
            print(f"RAG error: {line.strip()}")
    
    threading.Thread(target=read_stderr, daemon=True).start()
    
    return process

def start_audio_process():
    """Avvia il processo audio in un ambiente conda."""
    process = subprocess.Popen(
        ["conda", "run", "-n", "audio_env", "python", "audio_module.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Thread per leggere l'output
    def read_output():
        for line in process.stdout:
            try:
                data = json.loads(line)
                if "transcription" in data:
                    audio_to_rag_queue.put(data["transcription"])
            except json.JSONDecodeError:
                print(f"Audio output (non-JSON): {line.strip()}")
    
    threading.Thread(target=read_output, daemon=True).start()
    
    # Thread per leggere stderr
    def read_stderr():
        for line in process.stderr:
            print(f"Audio error: {line.strip()}")
    
    threading.Thread(target=read_stderr, daemon=True).start()
    
    return process

def main():
    print("Avvio del sistema di conversazione...")
    
    # Avvia i processi
    rag_process = start_rag_process()
    audio_process = start_audio_process()
    
    try:
        while True:
            # Gestisci input utente
            user_input = input("Premi 'r' per registrare, 'q' per uscire: ")
            
            if user_input.lower() == 'q':
                break
            
            if user_input.lower() == 'r':
                # Chiedi al modulo audio di registrare
                audio_process.stdin.write("record\n")
                audio_process.stdin.flush()
                
                # Attendi la trascrizione
                print("In attesa della trascrizione...")
                while audio_to_rag_queue.empty():
                    time.sleep(0.1)
                
                # Invia la query al RAG
                query = audio_to_rag_queue.get()
                print(f"Query: {query}")
                rag_process.stdin.write(f"{query}\n")
                rag_process.stdin.flush()
                
                # Attendi la risposta
                print("In attesa della risposta...")
                while rag_to_audio_queue.empty():
                    time.sleep(0.1)
                
                # Leggi la risposta e riproducila
                response = rag_to_audio_queue.get()
                print(f"Risposta: {response}")
                audio_process.stdin.write(f"speak:{response}\n")
                audio_process.stdin.flush()
    
    finally:
        # Termina i processi
        rag_process.stdin.write("EXIT\n")
        rag_process.stdin.flush()
        rag_process.terminate()
        
        audio_process.stdin.write("exit\n")
        audio_process.stdin.flush()
        audio_process.terminate()
        
        print("Sistema terminato.")

if __name__ == "__main__":
    main()