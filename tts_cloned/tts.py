#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from TTS.api import TTS
import torch
import sounddevice as sd
import numpy as np

class XTTS_UltraFast_Node:
    def __init__(self):
        # 1. Configurazione CUDA estrema
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)  # Riduce contention CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Caricamento modello ottimizzato
        rospy.loginfo("Caricamento modello XTTS ottimizzato...")
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(self.device)  # Spostamento esplicito su GPU

        # 3. Warm-up iper-aggressivo (5 micro-inferenze)
        rospy.loginfo("Warm-up turbo...")
        for _ in range(5):
            self.tts.tts(text=".", speaker_wav="voce_alessia.wav", language="it")
        
        # 4. Configurazione ROS low-latency
        self.sub = rospy.Subscriber(
            "tts_input",
            String,
            self.callback,
            queue_size=1,
            buff_size=2**18  # 256KB buffer
        )

    def callback(self, msg):
        try:
            # Generazione diretta senza controlli
            audio = self.tts.tts(
                text=msg.data,
                speaker_wav="voce_alessia.wav",
                language="it"
            )
            sd.play(np.array(audio), samplerate=22050, blocking=False)  # 16kHz per più velocità
            
        except Exception as e:
            rospy.logerr(f"Errore generazione: {str(e)}")

if __name__ == '__main__':
    rospy.init_node('xtts_ultra_fast')
    XTTS_UltraFast_Node()
    rospy.spin()