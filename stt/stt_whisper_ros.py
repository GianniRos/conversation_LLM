#!/usr/bin/env python

import rospy
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from std_msgs.msg import String

def audio_to_text_streaming():
    # Inizializza il nodo ROS
    rospy.init_node('audio_to_text_streaming_node', anonymous=True)

    # Crea il publisher per il testo trascritto
    pub = rospy.Publisher('transcribed_text', String, queue_size=10)

    # Inizializza il modello Whisper
    model = WhisperModel("base", device="cuda", compute_type="float16")

    # Impostazioni audio
    samplerate = 16000  # frequenza di campionamento
    block_duration = 0.5  # durata di ogni blocco in secondi
    buffer_duration = 5  # durata totale del buffer in secondi
    
    # Creazione coda e buffer
    audio_queue = queue.Queue()
    audio_buffer = np.zeros(shape=int(buffer_duration * samplerate), dtype=np.float32)
    
    # Flag per il controllo dello streaming
    streaming_active = True
    
    # Funzione di callback per la registrazione audio
    def audio_callback(indata, frames, time, status):
        if status:
            rospy.logwarn(f"Status: {status}")
        audio_queue.put(indata.copy())
    
    # Funzione per processare l'audio in streaming
    def process_audio():
        nonlocal streaming_active, audio_buffer
        temp_file = "temp_stream.wav"
        buffer_idx = 0
        
        while streaming_active and not rospy.is_shutdown():
            try:
                # Preleva il blocco audio dalla coda
                new_audio = audio_queue.get(timeout=1)
                new_audio = new_audio.flatten()
                
                # Aggiorna il buffer circolare
                next_idx = buffer_idx + len(new_audio)
                if next_idx <= len(audio_buffer):
                    audio_buffer[buffer_idx:next_idx] = new_audio
                else:
                    # Gestisce il wrap-around
                    overflow = next_idx - len(audio_buffer)
                    audio_buffer[buffer_idx:] = new_audio[:len(audio_buffer)-buffer_idx]
                    audio_buffer[:overflow] = new_audio[len(audio_buffer)-buffer_idx:]
                    
                buffer_idx = next_idx % len(audio_buffer)
                
                # Salva il buffer su file temporaneo
                import scipy.io.wavfile
                scipy.io.wavfile.write(temp_file, samplerate, audio_buffer)
                
                # Trascrizione dell'audio
                segments, _ = model.transcribe(
                    temp_file,
                    language="it", 
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,        # Aumenta questo valore (0.0-1.0) per richiedere maggiore certezza
                        min_speech_duration_ms=250,  # Aumenta per ignorare suoni brevi
                        max_speech_duration_s=float("inf"),
                        min_silence_duration_ms=100,  # Regola in base alle tue necessità
                        window_size_samples=1024,
                        speech_pad_ms=300
                    )
                )

                # Estrai il testo e pubblicalo su ROS
                transcribed_text = " ".join([seg.text for seg in segments])
                if transcribed_text.strip():  # Pubblica solo se c'è del testo
                    rospy.loginfo(f"Testo trascritto: {transcribed_text}")
                    pub.publish(transcribed_text)
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Errore durante la trascrizione: {e}")
    
    # Avvia il thread per il processing
    processing_thread = threading.Thread(target=process_audio)
    processing_thread.daemon = True
    processing_thread.start()
    
    try:
        # Avvia lo streaming audio
        rospy.loginfo("🎙️ Avvio streaming audio...")
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, 
                            blocksize=int(block_duration * samplerate), dtype='float32'):
            rospy.loginfo("✅ Streaming attivo. Premi Ctrl+C per terminare.")
            rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Interruzione richiesta dall'utente.")
    except Exception as e:
        rospy.logerr(f"Errore nello streaming audio: {e}")
    finally:
        # Pulizia
        streaming_active = False
        rospy.loginfo("Terminazione dello streaming audio.")

if __name__ == '__main__':
    try:
        audio_to_text_streaming()
    except rospy.ROSInterruptException:
        pass