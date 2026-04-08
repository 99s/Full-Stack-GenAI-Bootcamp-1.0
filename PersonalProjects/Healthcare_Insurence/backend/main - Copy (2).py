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

# ✅ FREE embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Paid but cheap LLM
from langchain_openai import ChatOpenAI

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# 🔐 Load ENV
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

# -------------------------------
# 🚀 FastAPI App
# -------------------------------
app = FastAPI(title="Policy Assistant (Cost Optimized RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 📄 File Loaders
# -------------------------------
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {}),
}

# -------------------------------
# 🧠 Initialize Knowledge Base
# -------------------------------
def initialize_knowledge_base():
    policy_dir = "./policies"

    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
        print("⚠️ Created /policies folder. Add files and restart.")
        return None

    all_docs = []

    for ext, (loader_cls, loader_kwargs) in LOADER_MAPPING.items():
        loader = DirectoryLoader(
            policy_dir,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs
        )
        try:
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext}: {e}")

    if not all_docs:
        print("⚠️ No documents found in /policies.")
        return None

    # Chunking
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
    )
    splits = splitter.split_documents(all_docs)

    print(f"📄 Total chunks: {len(splits)}")

    # ✅ FREE embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ CACHE VECTOR DB (IMPORTANT)
    if os.path.exists("./chroma_db"):
        print("✅ Loading existing vector DB (no cost)...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("⚡ Creating vector DB (one-time embedding cost)...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    return vectorstore.as_retriever(search_kwargs={"k": 6})


# -------------------------------
# 🚀 Initialize RAG
# -------------------------------
print("\n--- Initializing Policy RAG Engine ---")

retriever = initialize_knowledge_base()

if not retriever:
    print("❌ No documents loaded.")

# -------------------------------
# 🤖 LLM (Cheap OpenAI)
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -------------------------------
# 🧾 Prompt
# -------------------------------
# system_prompt = (
#     "You are an Intelligent Policy Assistant.\n"
#     "Answer ONLY from the given context.\n"
#     "If not found, say you don't know.\n"
#     "Always include sources.\n\n"
#     "Context: {context}"
# )
system_prompt = """
You are an Intelligent Policy Assistant.

STRICT RULES:
- Answer ONLY from provided context
- If context is empty → say "No relevant policy found"
- Always include source references
- Be precise and factual

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# -------------------------------
# 🔗 RAG Chain
# -------------------------------
rag_chain = None

if retriever:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# -------------------------------
# 🌐 API
# -------------------------------
@app.post("/chat")
async def chat_endpoint(payload: Dict[str, str]):

    if not rag_chain:
        raise HTTPException(
            status_code=500,
            detail="RAG not initialized. Add documents."
        )

    user_query = payload.get("text")

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Query text is required"
        )

    try:
        response = rag_chain.invoke({"input": user_query})

        sources = list(set([
            doc.metadata.get("source", "unknown")
            for doc in response.get("context", [])
        ]))

        return {
            "answer": response.get("answer"),
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI Engine Error: {str(e)}"
        )

# -------------------------------
# ▶️ Run
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)