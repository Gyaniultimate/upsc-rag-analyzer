## UPSC RAG Analyzer 📚🔍

A local AI tool that uses Retrieval-Augmented Generation (RAG) to cross-reference UPSC Prelims questions against Vision IAS Monthly Current Affairs PDFs. It determines if a question was "solvable" from the study material by using semantic search and cross-encoder reranking. 🚀

---

### Features
- **Semantic Search:** Finds answers even when exact keywords differ (e.g., matches "fiscal deficit" with "budgetary gap").
- **Precise Citations:** Outputs the exact PDF name and page number for every match.
- **Relevance Scoring:** Uses a cross-encoder model (`ms-marco-MiniLM`) to grade how well text answers a question (Score 0–10).
- **Privacy First:** Runs 100% locally — no data is sent to the cloud.

---

### Prerequisites
- **OS:** Windows, macOS, or Linux.
- **Python:** 3.10 or higher.
- **RAM:** 16 GB recommended (8 GB minimum).
- **GPU (Optional):** NVIDIA GPU recommended for speed (minutes on GPU vs hours on CPU).

---

### Project Structure
Before running, ensure your folders look exactly like this:

```
upsc-rag-analyzer/
├── data/
│   ├── pdfs/                 # Place your 12 Vision IAS PDF files here
│   └── questions.csv         # Your questions file (Must have headers: id, Question Text, option a, etc.)
├── db/                       # Empty folder (Database will be created here)
├── src/
│   ├── ingest.py             # Script 1: Reads PDFs & builds database
│   └── retrieve_and_score.py # Script 2: Finds answers for your questions
├── requirements.txt
└── README.md
```

Referenced files and folders:
- [data/pdfs/](data/pdfs/)
- [data/questions.csv](data/questions.csv)
- [db/](db/)
- [src/ingest.py](src/ingest.py)
- [src/retrieve_and_score.py](src/retrieve_and_score.py)

---

### Installation
Clone or download this repository and open a terminal in the project folder. Install required libraries:

```bash
pip install pandas langchain-community langchain-huggingface chromadb sentence-transformers pymupdf
```

**Note for GPU users:** install the GPU build of PyTorch for faster processing:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

### How to Run

#### Step 1: Ingest the Data (Reading Phase)
This script reads all PDFs in [data/pdfs/](data/pdfs/), cleans the text, and saves it into a local vector DB at [db/chroma_db/](db/chroma_db/).

```bash
python src/ingest.py
```

- Time estimate: GPU ~3–5 minutes, CPU ~45–60 minutes.
- Success message: "Vector DB saved successfully."

#### Step 2: Analyze Questions (Exam Phase)
This script reads [data/questions.csv](data/questions.csv), searches the DB for each question, reranks results, and writes analysis to `upsc_vision_analysis.csv`.

```bash
python src/retrieve_and_score.py
```

- Success message: "Analysis Complete! Results saved to upsc_vision_analysis.csv"

---

### Understanding the Output
Open `upsc_vision_analysis.csv`. Important columns:

- **Question_ID:** The ID from your input CSV.
- **Best_Match_Score:**
  - > 2.0: Strong Match (answer likely found)
  - 0 to 2.0: Possible Match (context similar)
  - < 0: No Match (topic likely not covered)
- **Source_PDF:** The exact PDF file where the supporting text was found.
- **Page_Num:** Page number in that PDF.
- **Extracted_Text:** The paragraph used as the supporting evidence.

---

### Troubleshooting
1. **CUDA Out Of Memory Error**
	- Cause: GPU ran out of VRAM.
	- Fix: Close other GPU-intensive apps or force CPU by setting `device='cpu'` in the scripts.

2. **KeyError: 'question_text'**
	- Cause: `questions.csv` headers don't match expected names.
	- Fix: Ensure CSV columns include `id` and `Question Text` (and options as required).

3. **An existing connection was forcibly closed**
	- Cause: Internet drop while downloading models.
	- Fix: Re-run the script — downloads will resume.

---

### Notes & Tips
- Keep your Vision IAS PDFs named clearly so `Source_PDF` citations are meaningful.
- If you plan to process many PDFs frequently, prefer a GPU to save time.
- Back up [db/chroma_db/](db/chroma_db/) after a successful ingest to avoid reprocessing.

---



