import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
 # Ensure ingest.py is in the same directory or adjust the import accordingly

# CONFIGURATION
DB_DIR = "./db/chroma_db"
QUESTIONS_FILE = "./data/questions.csv"
OUTPUT_FILE = "upsc_vision_analysis.csv"

# 1. RETRIEVAL MODEL (Same as ingest.py)
EMBED_MODEL_NAME = "BAAI/bge-m3" 

# 2. RERANKING MODEL (The "Judge")
# This model is specifically trained to say "How relevant is this text to this query?"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    # Load Resources
    print("Loading Vector DB on GPU...")
    # UPDATE THIS BLOCK
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cuda'} # <--- Add this here too
    )
    
    print("Loading Cross-Encoder on GPU...")
    # CrossEncoder usually detects GPU automatically, but you can force it:
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device='cuda')
    
    # ---------------------------------------------------------
    # 1. LOAD CSV & FIX HEADERS
    # ---------------------------------------------------------
    try:
        df_q = pd.read_csv(QUESTIONS_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Clean invisible spaces (e.g. "id " -> "id")
    df_q.columns = df_q.columns.str.strip()
    
    # Map YOUR headers to the script's variables
    # We rename them so the code below is standard
    header_map = {
        'Question Text': 'question_text',  # Renaming your header
        'id': 'id',                        # Keeping this as is
        'ID': 'id'                         # Just in case it's capitalized
    }
    df_q.rename(columns=header_map, inplace=True)
    
    # Validation check
    if 'question_text' not in df_q.columns:
        print(f"❌ Error: Column 'Question Text' not found.")
        print(f"   Detected columns: {list(df_q.columns)}")
        return

    print(f"Loaded {len(df_q)} questions.")
    
    results = []
    
    # ---------------------------------------------------------
    # 2. PROCESS LOOP
    # ---------------------------------------------------------
    for index, row in df_q.iterrows():
        q_id = row.get('id', index+1) 
        
        # A. BUILD QUERY: Combine Question + Options
        # This is CRITICAL for MCQs. The semantic answer is often in the options.
        q_text = row['question_text']
        
        # Collect all options into a single string
        # Using the exact headers you provided: "option a)", "option b)", etc.
        options_text = ""
        for opt_col in ['option (a)', 'option (b)', 'option (c)', 'option (d)']:
            if opt_col in row and pd.notna(row[opt_col]):
                options_text += f" {row[opt_col]}"
        
        # The full search query
        full_query = f"{q_text} {options_text}".strip()
        
        print(f"Processing Q{q_id}...", end="\r")
        vector_db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=hf_embeddings
    )
        # B. RETRIEVE (Vector Search)
        # Get top 5 candidates
        retrieved_docs = vector_db.similarity_search(full_query, k=5)
        
        if not retrieved_docs:
            print(f"Warning: No docs found for Q{q_id}")
            continue
            
        # C. RERANK (Cross-Encoder)
        # We compare the FULL query against the retrieved chunks
        pairs = [[full_query, doc.page_content] for doc in retrieved_docs]
        scores = cross_encoder.predict(pairs)
        
        # D. PICK WINNER
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

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\n✅ Analysis Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()