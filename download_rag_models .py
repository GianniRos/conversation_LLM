# Esegui con: conda run -n rag_env python download_rag_models.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings

# Directory di cache per i modelli
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

print("Scaricamento modelli LLM e embedding...")
tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    cache_dir=os.path.join(MODELS_CACHE_DIR, "llm_tokenizer")
)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    torch_dtype=torch.float16,
    load_in_4bit=True,
    cache_dir=os.path.join(MODELS_CACHE_DIR, "llm_model")
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.path.join(MODELS_CACHE_DIR, "embeddings")
)
print("Download completato per i modelli RAG!")