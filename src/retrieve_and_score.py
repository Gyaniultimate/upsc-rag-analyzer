import pandas as pd
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated import for recent versions
from sentence_transformers import CrossEncoder

# CONFIGURATION
DB_DIR = "./db/chroma_db"
QUESTIONS_FILE = "./data/questions.csv"
OUTPUT_FILE = "upsc_vision_analysis.csv"

# Model Config
EMBED_MODEL_NAME = "BAAI/bge-m3" 
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    # ---------------------------------------------------------
    # 1. SETUP RESOURCES (GPU & DB)
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    print("Loading Vector DB...")
    # Initialize Embedding Model (matches ingest.py)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': device}
    )
    
    # Connect to the existing Database
    vector_db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=hf_embeddings
    )
    
    print(f"Loading Cross-Encoder on {device}...")
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=device)
    
    # ---------------------------------------------------------
    # 2. LOAD CSV & PREPARE DATA
    # ---------------------------------------------------------
    print(f"Loading questions from {QUESTIONS_FILE}...")
    try:
        df_q = pd.read_csv(QUESTIONS_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Clean header names (remove hidden spaces)
    df_q.columns = df_q.columns.str.strip()
    
    # Standardize column names based on your file structure
    header_map = {
        'Question Text': 'question_text',
        'Question T': 'question_text', # Handle potential variations
        'id': 'id',
        'ID': 'id'
    }
    df_q.rename(columns=header_map, inplace=True)
    
    # Verify we have the critical column
    if 'question_text' not in df_q.columns:
        print(f"❌ Error: Column 'Question Text' not found.")
        print(f"   Found columns: {list(df_q.columns)}")
        return

    print(f"✅ Loaded {len(df_q)} questions.")
    
    results = []
    
    # ---------------------------------------------------------
    # 3. ANALYSIS LOOP
    # ---------------------------------------------------------
    for index, row in df_q.iterrows():
        q_id = row.get('id', index+1) 
        q_text = row['question_text']
        
        # Combine Question + Options for a rich semantic query
        options_text = ""
        # Loop through  specific option columns
        for opt_col in ['option (a)', 'option (b)', 'option (c)', 'option (d)']:
            # Check if column exists and value is not empty (NaN)
            if opt_col in row and pd.notna(row[opt_col]):
                options_text += f" {row[opt_col]}"
        
        full_query = f"{q_text} {options_text}".strip()
        
        print(f"Processing Q{q_id}...", end="\r")
        
        # A. RETRIEVE: Get Top 5 Candidates via Vector Search
     
        try:
            retrieved_docs = vector_db.similarity_search(full_query, k=5)
        except Exception as e:
            print(f"\n⚠️ Search failed for Q{q_id}: {e}")
            continue
        
        if not retrieved_docs:
            continue
            
        # B. RERANK: Use Cross-Encoder to find the precise answer
        pairs = [[full_query, doc.page_content] for doc in retrieved_docs]
        scores = cross_encoder.predict(pairs)
        
        # C. SELECT BEST MATCH
        # Zip docs and scores together, sort by score descending
        scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_docs[0]
        
        results.append({
            "Question_ID": q_id,
            "Question": q_text,
            "Best_Match_Score": float(best_score),
            "Source_PDF": best_doc.metadata.get("source", "Unknown"),
            "Page_Num": best_doc.metadata.get("page", "Unknown"),
            "Extracted_Text": best_doc.page_content
        })

    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\n✅ Analysis Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
