# 📊 Agente de Análise de Dados com RAG

Sistema inteligente para análise exploratória de dados (EDA) integrado com Retrieval Augmented Generation (RAG) para consulta em documentos técnicos. Permite análise estatística de datasets CSV combinada com consulta inteligente em documentos PDF.

## 🏗️ Estrutura do Projeto
```
eda_agent/
├── 📁 data/ # Dataset CSV (creditcard.csv)
├── 📁 rag_files/ # Documentos PDF para base de conhecimento
├── 📁 vectorstore/ # Banco vetorial persistido
├── 📁 venv/ # Ambiente virtual Python
├── 🔧 agent.py # Agente principal com ferramentas de análise
├── 🎨 app.py # Interface web Streamlit
├── 📥 dataset_download.py # Download automático do dataset
├── 🗄️ vectorstore_creator.py # Criador do banco vetorial
├── 🔍 vectorstore_loader.py # Carregador e testador do banco vetorial
├── 🔑 .env # Variáveis de ambiente (GROQ_API_KEY)
├── 📖 README.md # Documentação
└── .gitignore # Arquivos ignorados pelo Git
```


## 📁 Arquivos Principais

| Arquivo | Função |
|---------|--------|
| **`agent.py`** | Agente principal LangChain com ferramentas para estatísticas, visualização e RAG |
| **`app.py`** | Interface Streamlit com chat interativo e painel de controle |
| **`dataset_download.py`** | Download automático do dataset `creditcard.csv` da Kaggle |
| **`vectorstore_creator.py`** | Criação do banco vetorial a partir dos PDFs em `rag_files/` |
| **`vectorstore_loader.py`** | Carregamento e teste do banco vetorial persistido |
| **`.env`** | Configuração da API key do Groq |

## 🚀 Funcionalidades

### 🤖 Agente Inteligente (`agent.py`)
- **Análise Estatística**: Estatísticas descritivas, resumos de dataset, análise de colunas
- **Visualização de Dados**: Histogramas, boxplots e gráficos interativos
- **Sistema RAG**: Consulta inteligente em documentos PDF técnicos
- **Ferramentas Especializadas**:
  - `buscar_documentos()` - Busca semântica em base de conhecimento
  - `estatisticas_coluna()` - Análise estatística por coluna específica
  - `criar_histograma()` - Visualização de distribuição de dados
  - `criar_boxplot()` - Identificação de outliers e quartis
  - `explicar_conceito()` - Explicações baseadas em documentos
  - `nomes_colunas()` - Listagem de colunas disponíveis

### 🎨 Interface Web (`app.py`)
- **Chat Interativo**: Interface conversacional com histórico persistente
- **Visualização em Tempo Real**: Gráficos embutidos diretamente no chat
- **Painel de Controle Lateral**: Status do sistema, exemplos rápidos, visualização de dados
- **Design Responsivo**: Adaptável para diferentes dispositivos
- **Gerenciamento de Estado**: Memória de conversa e controle de sessão

### 📥 Gerenciamento de Dados (`dataset_download.py`)
- **Download Automático**: Dataset de fraudes em cartões de crédito da Kaggle
- **Pré-processamento Inteligente**: Amostragem automática para datasets grandes (>10k linhas)
- **Estrutura Organizada**: Armazenamento padronizado na pasta `data/`
- **Verificação de Integridade**: Confirmação de download e cópia bem-sucedida

### 🗄️ Sistema RAG
**`vectorstore_creator.py`**:
- **Processamento de PDFs**: Carregamento automático de documentos
- **Chunking Inteligente**: Divisão em trechos otimizados (600 chars, 100 overlap)
- **Embeddings**: Usa modelo `sentence-transformers/all-MiniLM-L6-v2`
- **Persistência**: Salva banco vetorial em `vectorstore/`

**`vectorstore_loader.py`**:
- **Carregamento Seguro**: Verificação de existência e integridade
- **Testes Automáticos**: Validação da qualidade das buscas
- **Estatísticas**: Contagem de documentos e verificação de funcionamento

## 🛠️ Configuração e Instalação

### Pré-requisitos
```bash
# Instalação das dependências
pip install streamlit pandas numpy matplotlib langchain groq chromadb sentence-transformers kagglehub python-dotenv

# Ou instale todas as dependências de uma vez
pip install -r requirements.txt
```
Configuração de Ambiente

Crie o arquivo .env:
```
GROQ_API_KEY=sua_chave_groq_aqui
```
Configure a Kaggle (para download do dataset):

Baixe kaggle.json da sua conta Kaggle

Coloque em ~/.kaggle/kaggle.json (Linux/Mac) ou C:\Users\seu_usuario\.kaggle\kaggle.json (Windows)

🎯 Fluxo de Uso
1. 🗄️ Preparar Base de Conhecimento (RAG)

# Coloque seus PDFs técnicos na pasta rag_files/
# Execute o criador do banco vetorial
```
python vectorstore_creator.py
```
2. 📥 Baixar Dataset
```
# Download automático do dataset de fraudes
python dataset_download.py
```
3. 🚀 Executar Sistema
```
# Interface web (recomendado)
streamlit run app.py

# Ou via console
python agent.py
```
4. 💬 Interagir com o Agente

Exemplos de Perguntas:

📊 Análise de Dados:

        "Resumo do dataset"

        "Estatísticas da coluna V1"

        "Criar histograma de Amount"

        "Quais são as colunas disponíveis?"

📚 Consulta em Documentos:

        "O que é desvio padrão?"

        "Explique o conceito de machine learning"

        "O que os documentos dizem sobre análise exploratória?"

🔧 Tecnologias Utilizadas

    LangChain: Framework para aplicações LLM com agents e tools

    Streamlit: Interface web interativa e responsiva

    Groq API: LLM de alta performance (Llama 3.1 8B Instant)

    ChromaDB: Banco de dados vetorial para RAG

    HuggingFace: Sentence transformers para embeddings

    Pandas/NumPy: Análise e processamento de dados

    Matplotlib: Visualização e geração de gráficos

    KaggleHub: Download automatizado de datasets

⚙️ Configurações Técnicas
Modelos e Parâmetros

    LLM: llama-3.1-8b-instant (Groq)

    Embeddings: sentence-transformers/all-MiniLM-L6-v2

    Chunk Size: 600 caracteres

    Chunk Overlap: 100 caracteres

    Retrieval: Top 3 documentos relevantes

Otimizações

    Amostragem: Datasets grandes são reduzidos para 10.000 linhas

    Memória: Buffer de conversa com limite de tokens

    Erros: Tratamento robusto de exceções e parsing errors

    Performance: Configurações para evitar limites de API


