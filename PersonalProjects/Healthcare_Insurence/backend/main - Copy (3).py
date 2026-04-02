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

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3"

app = FastAPI(title="Intelligent Policy Assistant (RAG)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 📄 Loaders
# -------------------------------
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {}),
}

# -------------------------------
# 🔁 Multi Query Expansion (NO LLM = FREE)
# -------------------------------
def expand_query(q):
    return [
        q,
        f"insurance coverage for {q}",
        f"policy clause about {q}",
        f"medical coverage related to {q}"
    ]

# -------------------------------
# 📄 Initialize Knowledge Base
# -------------------------------
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

def initialize_knowledge_base():
    persist_dir = "./chroma_db"

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=OLLAMA_HOST
    )

    # Try loading existing DB
    if os.path.exists(persist_dir):
        try:
            print("✅ Loading existing vector DB...")
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )

            # ✅ CHECK if DB actually has data
            if vectorstore._collection.count() > 0:
                print(f"✅ Loaded {vectorstore._collection.count()} vectors")
                return vectorstore
            else:
                print("⚠️ DB exists but empty. Rebuilding...")

        except Exception as e:
            print(f"⚠️ Failed loading DB: {e}")

    # ---------------------------
    # Build new DB
    # ---------------------------
    print("⚡ Creating new vector DB...")

    all_docs = []
    for ext, (loader_cls, loader_kwargs) in LOADER_MAPPING.items():
        loader = DirectoryLoader(
            "./policies",
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs
        )
        all_docs.extend(loader.load())

    if not all_docs:
        print("❌ No documents found in /policies")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = splitter.split_documents(all_docs)

    print(f"📄 Creating embeddings for {len(splits)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ Vector DB created and persisted")

    return vectorstore
# def initialize_knowledge_base():
#     policy_dir = "./policies"

#     if not os.path.exists(policy_dir):
#         os.makedirs(policy_dir)
#         print("⚠️ Created /policies folder. Add files and restart.")
#         return None

#     all_docs = []

#     for ext, (loader_cls, loader_kwargs) in LOADER_MAPPING.items():
#         loader = DirectoryLoader(
#             policy_dir,
#             glob=f"**/*{ext}",
#             loader_cls=loader_cls,
#             loader_kwargs=loader_kwargs
#         )
#         try:
#             all_docs.extend(loader.load())
#         except Exception as e:
#             print(f"Error loading {ext}: {e}")

#     if not all_docs:
#         print("⚠️ No documents found in /policies.")
#         return None

#     # ✅ Better chunking (CRITICAL FIX)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )
#     splits = splitter.split_documents(all_docs)

#     print(f"✅ Created {len(splits)} chunks")

#     # Embeddings (LOCAL = FREE)
#     embeddings = OllamaEmbeddings(
#         model="mxbai-embed-large",
#         base_url=OLLAMA_HOST
#     )

#     # Vector DB
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db"
#     )

#     return vectorstore


# -------------------------------
# 🚀 Init
# -------------------------------
print("\n--- Initializing Policy RAG Engine ---")

vectorstore = initialize_knowledge_base()

if not vectorstore:
    print("❌ No data loaded.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) if vectorstore else None

# LLM (LOCAL = FREE)
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_HOST,
    temperature=0
)

# ✅ Strong grounding prompt
system_prompt = """
You are an Intelligent Policy Assistant.

STRICT RULES:
- Answer ONLY from the provided context
- If answer is not found → say "No relevant policy found"
- Be precise and factual
- Always mention source file names if available

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

rag_chain = None
if retriever:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


# -------------------------------
# 🌐 API
# -------------------------------
@app.post("/chat")
async def chat_endpoint(payload: Dict[str, str]):

    if not retriever:
        raise HTTPException(500, "RAG not initialized")

    user_query = payload.get("text")
    if not user_query:
        raise HTTPException(400, "Query required")

    try:
        # ✅ Multi-query retrieval (BIG IMPROVEMENT)
        queries = expand_query(user_query)

        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        # Remove duplicates
        unique_docs = {d.page_content: d for d in all_docs}.values()

        # DEBUG (VERY IMPORTANT)
        print("\n🔍 Retrieved Chunks:")
        for d in list(unique_docs)[:3]:
            print(d.page_content[:200])
            print("------")

        # Run LLM
        response = rag_chain.invoke({
            "input": user_query,
            "context": list(unique_docs)
        })

        sources = list(set([
            d.metadata.get("source", "unknown")
            for d in unique_docs
        ]))

        return {
            "answer": response.get("answer"),
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# -------------------------------
# ▶️ Run
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)