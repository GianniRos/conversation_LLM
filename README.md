# conversation_LLM# Sistema di Conversazione RAG con Interfaccia Vocale

Questo sistema consente di interagire con un modello di linguaggio potenziato da RAG (Retrieval-Augmented Generation) utilizzando un'interfaccia vocale. Funziona completamente offline dopo il download iniziale dei modelli.

---

## Prerequisiti

- **Python** 3.10
- **Conda**
- Supporto audio (microfono e altoparlanti)
- Spazio su disco (circa 8-10 GB per i modelli)

---

## Configurazione degli Ambienti

### Ambiente Audio
Crea un nuovo ambiente con Python 3.10 e installa le dipendenze nell'ordine corretto:

```bash
conda create -n audio_env python=3.10
conda activate audio_env

pip install numpy==1.22.0
pip install numexpr==2.8.4
pip install TTS==0.22.0
pip install torch==2.1
pip install transformers==4.33.0
pip install sounddevice==0.4.6
pip install soundfile==0.12.1
```

---

## Download dei Modelli per Uso Offline

Prima di utilizzare il sistema offline, scarica i modelli necessari. Assicurati di avere una connessione internet attiva per questa fase.

### Modelli RAG
```bash
conda run -n rag_env python download_rag_models.py
```

### Modelli Audio
```bash
conda run -n audio_env python download_audio_models.py
```

---

## Utilizzo del Sistema

Una volta configurati gli ambienti e scaricati i modelli, puoi eseguire il sistema:

```bash
python main.py
```

### Comandi Disponibili

- **r**: Avvia la registrazione audio (registra per 5 secondi)
- **q**: Termina il programma

---

## Struttura del Progetto

- **main.py**: Script principale che coordina i moduli RAG e audio
- **rag_module.py**: Modulo per il recupero delle informazioni e la generazione delle risposte
- **audio_module.py**: Modulo per il riconoscimento vocale e la sintesi della voce
- **knowledge_base/**: Directory contenente i documenti di conoscenza per il sistema RAG
- **models_cache/**: Directory dove vengono salvati i modelli scaricati

---

## Knowledge Base

Per migliorare le risposte del sistema, puoi aggiungere documenti di testo alla directory `knowledge_base/`. Ogni file `.txt` verrà automaticamente indicizzato e utilizzato per arricchire le risposte.

---

## Risoluzione dei Problemi

- **Errori di compatibilità**: Assicurati di installare le librerie nell'ordine specificato.
- **Problemi audio**: Verifica che microfono e altoparlanti siano configurati correttamente.
- **Prestazioni lente**: Considera l'uso di una GPU per accelerare l'elaborazione del modello LLM.

---

## Note

- Il sistema utilizza la GPU se disponibile; altrimenti, opererà sulla CPU (più lento).
- La prima esecuzione potrebbe richiedere più tempo per caricare i modelli in memoria.
- Il modello LLM utilizzato è **Mistral-7B-Instruct**, quantizzato a 4-bit per ridurre l'utilizzo di memoria.
- Questo sistema è progettato per funzionare completamente offline dopo il download iniziale dei modelli. Assicurati di eseguire gli script di download una sola volta con connessione internet attiva.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>
<p>
  :man: <b>Giovanni Rosato</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>giovanni.rosato@iit.it</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/GianniRos">@GianniRos</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linkedin: <a href="https://www.linkedin.com/in/giovanni-rosato-6284bb161/">@giovanni-rosato-linkedin</a> <br>
</p>