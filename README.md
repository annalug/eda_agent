# 📊 Data Analysis Agent with RAG

An intelligent system for Exploratory Data Analysis (EDA) integrated with Retrieval-Augmented Generation (RAG) for querying technical documents.
It performs statistical analysis on CSV datasets while enabling semantic search over PDF files.
## 🏗️ Project Structure
```
eda_agent/
├── 📁 data/                 # CSV dataset (creditcard.csv)
├── 📁 rag_files/            # PDF documents used as knowledge base
├── 📁 vectorstore/          # Persisted vector database
├── 📁 venv/                 # Python virtual environment
├── 🔧 agent.py              # Main LangChain agent with tools (EDA + RAG)
├── 🎨 app.py                # Streamlit web interface
├── 📥 dataset_download.py   # Automated dataset downloader
├── 🗄️ vectorstore_creator.py # Vectorstore creation from PDFs
├── 🔍 vectorstore_loader.py  # Vectorstore loader/tester
├── 🔑 .env                  # Environment variables (GROQ_API_KEY)
└── 📖 README.md             # Documentation
```


## 📁 Main Files

| Arquivo | Função |
|---------|--------|
| **`agent.py`** | LangChain agent with statistical tools, visualizations, and RAG search |
| **`app.py`** | Streamlit interface with chat, real-time plots, and control panel |
| **`dataset_download.py`** | Automatic download of creditcard.csv from Kaggle via KaggleHub |
| **`vectorstore_creator.py`** | PDF loader, text splitter, embeddings, and ChromaDB vectorstore creation |
| **`vectorstore_loader.py`** | Safe loader and tester for the persisted vectorstore |
| **`.env`** | Contains the Groq API key |

## 🚀 Features

### 🤖 Intelligent Agent (`agent.py`)
- **Statistical Analysis**: Descriptive statistics, dataset summary, numerical analysis
- **Data Visualization**: Histograms and boxplots generated on demand
- **RAG System**: Semantic search across technical PDF documents
- **Specialized Tools**:
  - `buscar_documentos()` - semantic PDF retrieval
  - `estatisticas_coluna()` - column-level statistics
  - `criar_histograma()` - distribution visualization
  - `criar_boxplot()` - outlier & quartile inspection
  - `explicar_conceito()` - list available dataset columns
  - `nomes_colunas()` - global numerical summary

### 🎨 Web Interface (`app.py`)
- **Interactive Chat**: with persistent history
- **Real-Time Visualizations**: embedded directly in the chat
- **Sidebar Control Panel**: showing system status and dataset details
- **Responsive Design**: for desktop/tablet
- **State Management**: for conversation and session control

### 📥 Data Handling (`dataset_download.py`)
- **Automatic Download**: of the credit card fraud dataset
- **Smart Pre-processing**: automatic sampling for large datasets (>10k rows)
- **Consistent Directory Structure**: under `data/`
- **Integrity Check**: to confirm successful download and copy

### 🗄️ RAG System
**`vectorstore_creator.py`**:
- Loads all PDFs from
- Splits text into optimized chunks (600 chars, 100 overlap)
- Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Saves a fully persistent ChromaDB `vectorstore/`

**`vectorstore_loader.py`**:
- Safely loads the vectorstore
- Performs integrity checks and quick retrieval tests
- Displays document counts and retrieval quality

## 🛠️ Installation & Setup

### Requirements
```bash

pip install streamlit pandas numpy matplotlib langchain groq chromadb sentence-transformers kagglehub python-dotenv


pip install -r requirements.txt
```
Environment Setup

Create your .env:
```
GROQ_API_KEY=sua_chave_groq_aqui
```
Configure Kaggle (dataset download):

Download your kaggle.json 

Place it in: ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\seu_usuario\.kaggle\kaggle.json (Windows)

🎯 Usage Workflow:

1. 🗄️ Prepare the Knowledge Base (RAG)

Add your PDFs to rag_files/ and run:

```
python vectorstore_creator.py
```
2. 📥 Download the Dataset
```
# Download automático do dataset de fraudes
python dataset_download.py
```
3. 🚀 Run the System
```
# Web interface (recommended)
streamlit run app.py

# or via console
python agent.py
```
4. 💬 Interact with the Agent

📊 Data Analysis Examples:Data Analysis Examples:

    “Dataset summary”
    
    “Statistics for column V1”
    
    “Create histogram of Amount”
    
    “Which columns are available?”

📚 Document Retrieval Examples:

    “What is standard deviation?”
    
    “Explain machine learning”
    
    “What do the documents say about exploratory analysis?”

🔧 Technologies Used

    LangChain — LLM agents and tooling
    
    Streamlit — interactive and responsive UI
    
    Groq API (Llama 3.1 8B Instant) — ultra-fast LLM inference
    
    ChromaDB — vector database for RAG
    
    HuggingFace — Sentence Transformers for embeddings
    
    Pandas / NumPy — data processing
    
    Matplotlib — visualizations
    
    KaggleHub — automated dataset download

⚙️ Technical Configuration

    LLM: llama-3.1-8b-instant (Groq)
    Embeddings: sentence-transformers/all-MiniLM-L6-v2
    Chunk Size: 600 characters
    Chunk Overlap: 100 characters
    Retrieval: Top 3 most relevant documents
    
    Optimizations
    
    Automatic sampling for large datasets
    
    Conversation buffer with token limits
    
    Robust error handling
    
    Performance configurations to avoid API constraints


