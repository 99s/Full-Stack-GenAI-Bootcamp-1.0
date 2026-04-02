import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# LangChain Imports
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader, 
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# Updated imports for modern LangChain
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# Use these exact imports - they are the most robust across recent versions
# from langchain.chains.retrieval import create_retrieval_chain
# Change from: from langchain.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

# 1. Load Environment Variables
load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
# Model check: Ensure your cloud provider supports this tag
CLOUD_MODEL = "llama3:70b-cloud" 

app = FastAPI(title="Intelligent Policy Assistant (RAG)")

# 2. Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Document Loading Logic
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {}),
}

def initialize_knowledge_base():
    policy_dir = "./policies"
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
        print(f"Created {policy_dir} folder. Add your policy files (PDF, XLSX, etc.) here.")
        return None

    all_docs = []
    for ext, (loader_cls, loader_kwargs) in LOADER_MAPPING.items():
        loader = DirectoryLoader(
            policy_dir, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls, 
            loader_kwargs=loader_kwargs
        )
        all_docs.extend(loader.load())

    if not all_docs:
        print("No documents found in /policies.")
        return None

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Cloud Embeddings via Ollama Cloud
    # embeddings = OllamaEmbeddings(
    #     model="mxbai-embed-large",
    #     base_url=OLLAMA_HOST,
    #     headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    # )
    embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_HOST
    )

    # Vector Storage
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Global RAG Setup
print("--- Initializing Policy RAG Engine ---")
retriever = initialize_knowledge_base()

# Initialize Cloud-based LLM
llm = ChatOllama(
    model=CLOUD_MODEL,
    base_url=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    temperature=0
)

# Professional Prompting
system_prompt = (
    "You are an Intelligent Policy Assistant. Use the provided context "
    "to answer the question. If the answer is not in the context, say "
    "you do not have enough information. Always list the source files used.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain) if retriever else None

# 5. API Routes
@app.post("/chat")
async def chat_endpoint(payload: Dict[str, str]):
    if not rag_chain:
        print('error : not rag_chain-Backend data not initialized')
        raise HTTPException(status_code=500, detail="Backend data not initialized.")
    
    user_query = payload.get("text")
    if not user_query:
        print('error : Query text is required.')
        raise HTTPException(status_code=400, detail="Query text is required.")

    try:
        # LangSmith tracing is handled by environment variables
        response = rag_chain.invoke({"input": user_query})
        
        # Extract unique source names
        sources = list(set([doc.metadata.get("source") for doc in response["context"]]))
        
        return {
            "answer": response["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)