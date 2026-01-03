import textwrap
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# CONFIGURATION (Must match ingest.py exactly)
DB_DIR = "./db/chroma_db"
# If you used the local path in ingest.py, use it here too!
# Otherwise use "BAAI/bge-m3" or "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "BAAI/bge-m3" 

def inspect():
    print(f"Loading database from {DB_DIR}...")
    
    # 1. Initialize Embedding Function
    # (Chroma needs this to understand the vectors, even just to load the DB properly)
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 2. Load the Database
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_fn)
    
    # ---------------------------------------------------------
    # CHECK 1: Volume Check
    # ---------------------------------------------------------
    # This accesses the underlying collection count directly
    count = vector_db._collection.count()
    print(f"\n✅ Total Chunks Stored: {count}")
    
    if count == 0:
        print("❌ ERROR: Database is empty! Check ingest.py again.")
        return

    # ---------------------------------------------------------
    # CHECK 2: Raw Data Peek (The "Sanity" Check)
    # ---------------------------------------------------------
    print(f"\n🔍 Peeking at first 3 chunks to verify Metadata & Cleaning:")
    
    # Get first 3 docs
    data = vector_db.get(limit=3)
    
    for i in range(3):
        meta = data['metadatas'][i]
        text = data['documents'][i]
        
        print(f"\n--- Chunk {i+1} ---")
        print(f"📄 Source: {meta.get('source')} | Page: {meta.get('page')}")
        print(f"📝 Text Snippet: {text[:200]}...") # Printing first 200 chars
        print("-" * 50)

    # ---------------------------------------------------------
    # CHECK 3: Functional Search Test
    # ---------------------------------------------------------
    print(f"\n🧪 Running a Test Query: 'Election Commission of India'")
    results = vector_db.similarity_search_with_score("Election Commission of India", k=1)
    
    if results:
        doc, score = results[0]
        print(f"\n✅ Match Found!")
        print(f"   Score: {score:.4f} (Lower is closer distance)")
        print(f"   Source: {doc.metadata['source']} (Page {doc.metadata['page']})")
        print(f"   Content: {textwrap.shorten(doc.page_content, width=150)}")
    else:
        print("❌ No results found. Something is wrong with the embeddings.")

if __name__ == "__main__":
    inspect()