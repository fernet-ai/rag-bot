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
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 3: Genera embeddings dai chunks e costruisci il database vettoriale
        self._build_vector_store()

        # Step 4: Inizializza un LLM per la generazione di risposte
        self.generator = pipeline("text-generation", model="distilgpt2")

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


    def run(self, query: str, k: int = 3, threshold: float = 0.3) -> str:
        self.status = "Processing"
        
        # Step 1: Retrieve dei documenti più rilevanti dal db vettoriale rispetto alla query
        results = self.db.similarity_search_with_score(query, k=k)

        print(f"Risultati della ricerca: {results}")

        # # Step 2: Filtra quelli sopra la soglia
        # filtered = [(doc, score) for doc, score in results if score <= threshold]

        # if not filtered:
        #     return "Non sono riuscito a trovare informazioni sufficientemente rilevanti."

        # Step 3: Costruisci il contesto da passare al prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {query}
        """)
        prompt = prompt_template.format(context=context_text, query=query)

        # Step 4: Genera la risposta con il LLM
        response = self.generator(prompt, max_length=100, do_sample=True)
        response_text = response[0]['generated_text']

        # Step 5: Prepara la risposta formattata con le fonti
        sources = [doc.metadata.get("source", None) for doc, _ in results]
        formatted = f"Risposta: {response_text.strip()}\n\nFonti: {sources}"
        return formatted





    def retrieve_relevant_docs(self, query: str, k: int = 5, threshold: float = 0.5) -> dict:
        self.status = "Processing"

        # Recupera i documenti più rilevanti dal db vettoriale
        results = self.db.similarity_search_with_score(query, k=k)

        # Filtra solo quelli che superano la soglia di "accettabilità"
        filtered = [
            {
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc, score in results
            if score <= threshold
        ]

        self.status = "Ready"

        if not filtered:
            return {
                "query": query,
                "results": [],
                "message": "Nessun documento rilevante trovato.",
                "threshold": threshold
            }

        return {
            "query": query,
            "results": filtered,
            "count": len(filtered),
            "threshold": threshold
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


