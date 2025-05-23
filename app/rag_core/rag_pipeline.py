import os
from pathlib import Path
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline


class RAGPipeline:
    def __init__(self, data_path: str = None, chroma_path: str = None):     
        
        base_dir = Path(__file__).resolve().parent.parent  # ossia app/
        self.data_path = data_path or str(base_dir / "data")
        self.chroma_path = chroma_path or str(base_dir / "chroma")
        self.status = "Not ready"

        print("Inizio creazione della pipeline RAG ... 🔄")

        # Step 1: Carica e splitta in chunks i documenti nella cartella /data
        self._load_and_split_documents()

        # Step 2: Inizializza un modello di embedding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="nickprock/sentence-bert-base-italian-uncased" # oppure "sentence-transformers/all-MiniLM-L6-v2"
            
        )

        # Step 3: Genera embeddings dai chunks e costruisci il database vettoriale
        self._build_vector_store()

        # Step 4: Inizializza un LLM per la generazione di risposte
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

        print("Pipeline RAG pronta. ✅")
        self.status = "Ready"




    def _load_and_split_documents(self):
        loader = DirectoryLoader(self.data_path, glob="**/*.md", recursive=True)
        self.documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        self.chunks = splitter.split_documents(self.documents)
        print(f"Split {len(self.documents)} documents into {len(self.chunks)} chunks.")



    def _build_vector_store(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        self.db = Chroma.from_documents(
            self.chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_path,
        )
        # self.db.persist() # Con le versioni recenti di Chroma (>= 0.4.x), la persistenza è automatica: ogni volta che inserisci documenti nel DB, Chroma li salva già sul disco


    def run(self, query_text: str, k: int = 3, threshold: float = 0.4) -> dict:
        self.status = "Processing"
        
        # Step 1: Retrieve dei documenti più rilevanti dal db vettoriale rispetto alla query
        results = self.db.similarity_search_with_score(query_text, k=k)

        # Step 2: Filtra quelli sopra la soglia
        filtered = [(doc, score) for doc, score in results if score >= threshold]

        if not filtered:
            return {
                "answer": None,
                "sources": [],
                "message": "Nessun documento sufficientemente rilevante trovato.",
                "status": "no_relevant_results"
            }
        
        # Step 3: Prepara il prompt e costruisci il contesto da passare al prompt
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {query}
        """

        # Unisci i contenuti testuali dei documenti trovati
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered])

        # Crea il template e formatta il prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, query=query_text)

        # Step 4: Genera la risposta con il LLM
        response = self.generator(prompt, max_new_tokens=100, do_sample=True)
        response_text = response[0]['generated_text'].strip()
        print(f"Risposta generata: {response_text}")

        # Step 5: Prepara la risposta formattata con le fonti
        sources = [doc.metadata.get("source", None) for doc, _ in filtered]

        return {
            "answer": response_text,
            "sources": sources,
            "query": query_text,
            "relevant_documents": len(filtered),
            "status": "ok"
        }





    def retrieve_relevant_docs(self, query: str, k: int = 3, threshold: float = 0.4) -> dict:
        self.status = "Processing"

        # Recupera i documenti con punteggi di rilevanza (score alto = meglio)
        results = self.db.similarity_search_with_score(query, k=k)

        # Filtro basato su soglia di rilevanza
        filtered = [
            {
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc, score in results
            if score >= threshold
        ]

        self.status = "Ready"

        if not filtered:
            return {
                "query": query,
                "results": [],
                "message": "Nessun documento sufficientemente rilevante trovato.",
                "threshold": threshold,
                "status": "no_relevant_results"
            }

        return {
            "query": query,
            "results": filtered,
            "count": len(filtered),
            "threshold": threshold,
            "status": "ok"
        }






    def info(self):
        return {
            "name": "RAGPipeline",
            "description": "A LangChain-based RAG pipeline using HuggingFace + Chroma.",
            "status": self.status,
            "version": "1.0.0",
            "documents_loaded": len(self.documents),
            "chunks_created": len(self.chunks),
        }


