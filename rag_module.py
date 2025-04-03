import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Directory di cache per i modelli
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# Configurazione GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class RAGProcessor:
    def __init__(self, knowledge_base_path, model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Carica il modello LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=os.path.join(MODELS_CACHE_DIR, "llm_tokenizer"),
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_4bit=True,
            cache_dir=os.path.join(MODELS_CACHE_DIR, "llm_model"),
            local_files_only=True,
        )
        
        # Carica il modello di embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device, 'cache_folder': os.path.join(MODELS_CACHE_DIR, "embeddings")},
            cache_folder=os.path.join(MODELS_CACHE_DIR, "embeddings"),
        )
        
        # Carica la knowledge base
        self.load_knowledge_base(knowledge_base_path)
        
    def load_knowledge_base(self, path):
        # Carica i documenti
        documents = []
        for file in os.listdir(path):
            if file.endswith('.txt'):
                loader = TextLoader(os.path.join(path, file))
                documents.extend(loader.load())
        
        # Dividi i documenti in chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Crea il database vettoriale
        self.vectordb = FAISS.from_documents(texts, self.embeddings)
    
    def retrieve_relevant_context(self, query, k=3):
        docs = self.vectordb.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def generate_response(self, query):
        # Recupera il contesto rilevante
        context = self.retrieve_relevant_context(query)
        
        # Prepara il prompt
        prompt = f"""
        Di seguito sono riportate alcune informazioni rilevanti:
        {context}
        
        Basandoti solo sulle informazioni fornite, rispondi alla seguente domanda:
        {query}
        """
        
        # Genera la risposta
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Estrai solo la risposta
        response = response.split(query)[-1].strip()
        return response

    def process_query(self, query):
        return self.generate_response(query)

# Server API semplice
if __name__ == "__main__":
    import sys
    import time
    
    # Percorso alla directory contenente i file della knowledge base
    kb_path = "./knowledge_base"
    
    # Crea la directory se non esiste
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        with open(os.path.join(kb_path, "example.txt"), "w") as f:
            f.write("Questo è un esempio di conoscenza che il modello può utilizzare per rispondere alle domande.")
    
    print("Inizializzazione del sistema RAG...")
    rag_processor = RAGProcessor(kb_path)
    print("Sistema RAG pronto. In attesa di query...")
    
    # Attendi query da stdin
    for line in sys.stdin:
        query = line.strip()
        if query == "EXIT":
            break
        
        response = rag_processor.process_query(query)
        
        # Scrivi la risposta a stdout
        print(json.dumps({"response": response}))
        sys.stdout.flush()