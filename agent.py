# agent_com_rag.py - Agente com RAG integrado

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_core.messages import SystemMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class AgenteEDAComRAG:
    """
    Agente de Análise de Dados com suporte a RAG (Retrieval Augmented Generation)
    """

    def __init__(self):
        """Inicializa o agente"""
        self.llm = None
        self.df = None
        self.arquivo_carregado = None
        self.arquivos_disponiveis = []
        self.vectorstore = None
        self.retriever = None

        print("🚀 Inicializando Agente com RAG...")
        self._carregar_configuracoes()
        self._carregar_vectorstore()
        self._carregar_arquivos_automaticamente()

    def _carregar_configuracoes(self):
        """Carrega variáveis de ambiente e inicializa o LLM"""
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY não encontrada. Configure no arquivo .env")

        # Modelos disponíveis no Groq:
        # - llama-3.3-70b-versatile (Recomendado - mais inteligente, MAS menor limite)
        # - llama-3.1-8b-instant (Mais rápido e maior limite)
        # - mixtral-8x7b-32768 (Ótimo para contexto longo)
        # - gemma2-9b-it (Equilibrado)

        model_name = "llama-3.1-8b-instant"  # Modelo mais rápido com maior limite

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name=model_name
        )
        print(f"✅ LLM configurado: {model_name}")

    def _carregar_vectorstore(self):
        """Carrega o banco de dados vetorial para RAG"""
        vectorstore_dir = "vectorstore"

        if not Path(vectorstore_dir).exists():
            print("⚠️  Pasta 'vectorstore' não encontrada. RAG desabilitado.")
            print("💡 Execute vectorstore_creator.py primeiro para criar o banco vetorial")
            return

        try:
            print("📚 Carregando banco vetorial...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            self.vectorstore = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings
            )

            # Criar retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retorna top 3 resultados
            )

            # Verificar se há documentos
            count = self.vectorstore._collection.count()
            if count > 0:
                print(f"✅ RAG ativado! {count} chunks disponíveis para consulta")
            else:
                print("⚠️  Vectorstore vazio. Adicione PDFs e recrie o banco.")
                self.vectorstore = None
                self.retriever = None

        except Exception as e:
            print(f"⚠️  Erro ao carregar vectorstore: {e}")
            print("💡 RAG desabilitado. Apenas análise de CSV disponível.")
            self.vectorstore = None
            self.retriever = None

    def _carregar_arquivos_automaticamente(self):
        """Carrega automaticamente arquivos CSV da pasta 'data'"""
        pasta = "data"
        if not os.path.exists(pasta):
            print(f"⚠️  Pasta '{pasta}' não existe. Criando...")
            os.makedirs(pasta, exist_ok=True)
            return

        self.arquivos_disponiveis = glob.glob(os.path.join(pasta, "*.csv"))

        if self.arquivos_disponiveis:
            print(f"📁 Encontrados {len(self.arquivos_disponiveis)} arquivo(s) CSV:")
            for i, arquivo in enumerate(self.arquivos_disponiveis, 1):
                nome = os.path.basename(arquivo)
                tamanho = os.path.getsize(arquivo)
                print(f"   {i}. {nome} ({tamanho:,} bytes)")
            self._carregar_primeiro_arquivo()
        else:
            print(f"📁 Nenhum CSV encontrado em '{pasta}'")

    def _carregar_primeiro_arquivo(self):
        """Carrega o primeiro arquivo CSV"""
        primeiro_arquivo = self.arquivos_disponiveis[0]
        nome = os.path.basename(primeiro_arquivo)
        print(f"⏳ Carregando '{nome}'...")

        try:
            self.df = pd.read_csv(primeiro_arquivo)
            self.arquivo_carregado = primeiro_arquivo
            print(f"✅ Arquivo carregado: {self.df.shape[0]:,} linhas × {self.df.shape[1]} colunas")

            if len(self.df) > 10000:
                print("📊 Dataset grande. Usando amostra de 10.000 linhas.")
                self.df = self.df.sample(n=10000, random_state=42)
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            self.df = None

    def tem_dataset_carregado(self):
        """Verifica se há dataset carregado"""
        return self.df is not None and not self.df.empty

    def tem_rag_disponivel(self):
        """Verifica se o RAG está disponível"""
        return self.retriever is not None

    # ==================== FERRAMENTAS DO AGENTE ====================

    def buscar_documentos(self, pergunta: str) -> str:
        """
        Busca informações nos documentos PDF indexados usando RAG.
        Use esta ferramenta para responder perguntas sobre documentos técnicos, artigos, relatórios, etc.

        Args:
            pergunta: A pergunta ou termo de busca
        """
        if not self.tem_rag_disponivel():
            return "❌ RAG não está disponível. Execute vectorstore_creator.py primeiro."

        if not pergunta:
            return "❌ Forneça uma pergunta para buscar nos documentos."

        try:
            # Buscar documentos relevantes
            docs = self.retriever.get_relevant_documents(pergunta)

            if not docs:
                return "📭 Nenhum documento relevante encontrado."

            # Formatar resultados
            resultado = f"📚 Encontrei {len(docs)} trechos relevantes:\n\n"

            for i, doc in enumerate(docs, 1):
                conteudo = doc.page_content[:300]  # Limitar tamanho
                fonte = doc.metadata.get('source', 'Desconhecida')
                pagina = doc.metadata.get('page', '?')

                resultado += f"🔹 Trecho {i} (Fonte: {Path(fonte).name}, Pág: {pagina}):\n"
                resultado += f"{conteudo}...\n\n"

            return resultado

        except Exception as e:
            return f"❌ Erro na busca: {str(e)}"

    def resumo_dataset(self, query: str = "", *args, **kwargs) -> str:
        """Retorna resumo do dataset CSV carregado. O parâmetro query é ignorado."""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset CSV carregado."

        nome = os.path.basename(self.arquivo_carregado)
        return (f"📊 Dataset: {nome}\n"
                f"📏 Dimensões: {self.df.shape[0]:,} linhas × {self.df.shape[1]} colunas\n"
                f"📋 Colunas: {', '.join(self.df.columns[:5])}...")

    def nomes_colunas(self, query: str = "", *args, **kwargs) -> str:
        """Retorna nomes das colunas do dataset. O parâmetro query é ignorado."""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."

        colunas = list(self.df.columns)
        if len(colunas) > 15:
            return f"📋 {len(colunas)} colunas: {', '.join(colunas[:10])}..."
        return f"📋 Colunas: {', '.join(colunas)}"

    def estatisticas_coluna(self, coluna: str) -> str:
        """Calcula estatísticas descritivas para uma coluna"""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."
        if coluna not in self.df.columns:
            return f"❌ Coluna '{coluna}' não encontrada."

        if pd.api.types.is_numeric_dtype(self.df[coluna]):
            stats = self.df[coluna].describe()
            return (f"📊 Estatísticas '{coluna}':\n"
                    f"  • Média: {stats['mean']:.2f}\n"
                    f"  • Desvio: {stats['std']:.2f}\n"
                    f"  • Mín: {stats['min']:.2f} | Máx: {stats['max']:.2f}\n"
                    f"  • Mediana: {stats['50%']:.2f}")
        else:
            unique = self.df[coluna].nunique()
            top = self.df[coluna].value_counts().head(3)
            return (f"📝 Estatísticas '{coluna}':\n"
                    f"  • Valores únicos: {unique}\n"
                    f"  • Top 3:\n{top.to_string()}")

    def criar_histograma(self, coluna: str) -> str:
        """Cria um histograma para uma coluna numérica do dataset"""
        return self.criar_grafico("histograma", coluna)

    def criar_boxplot(self, coluna: str) -> str:
        """Cria um boxplot para uma coluna numérica do dataset"""
        return self.criar_grafico("boxplot", coluna)

    def criar_grafico(self, tipo: str, coluna: str) -> str:
        """Cria gráfico (histograma ou boxplot) para coluna numérica"""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."
        if coluna not in self.df.columns:
            return f"❌ Coluna '{coluna}' não encontrada."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return f"❌ '{coluna}' não é numérica."

        plt.figure(figsize=(10, 6))

        try:
            if tipo.lower() == "histograma":
                plt.hist(self.df[coluna].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Histograma - {coluna}')
                plt.xlabel(coluna)
                plt.ylabel('Frequência')
            elif tipo.lower() == "boxplot":
                plt.boxplot(self.df[coluna].dropna())
                plt.title(f'Boxplot - {coluna}')
                plt.ylabel(coluna)
            else:
                return "❌ Tipo inválido. Use 'histograma' ou 'boxplot'."

            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"📈 Gráfico criado!\n![Gráfico](data:image/png;base64,{img_base64})"

        except Exception as e:
            plt.close()
            return f"❌ Erro: {str(e)}"

    def ajuda(self, query: str = "", *args, **kwargs) -> str:
        """Mostra exemplos de perguntas. O parâmetro query é ignorado."""
        rag_status = "✅ Ativo" if self.tem_rag_disponivel() else "❌ Desativado"
        csv_status = "✅ Carregado" if self.tem_dataset_carregado() else "❌ Não carregado"

        return f"""
💡 EXEMPLOS DE PERGUNTAS:

📚 Consulta em PDFs (RAG {rag_status}):
- "o que os documentos dizem sobre machine learning?"
- "explique o conceito X baseado nos PDFs"
- "resumo sobre tecnologia Y"

📊 Análise de CSV ({csv_status}):
- "resumo do dataset"
- "nomes das colunas"
- "estatísticas da coluna Amount"
- "criar histograma de Amount"
- "criar boxplot de V1"
        """

    # ==================== CRIAÇÃO DO EXECUTOR ====================

    def criar_executor(self) -> AgentExecutor:
        """Cria o AgentExecutor com todas as ferramentas"""

        # Criar ferramentas manualmente (sem decorator @tool)
        ferramentas = [
            Tool(
                name="buscar_documentos",
                func=self.buscar_documentos,
                description="Busca informações nos documentos PDF indexados. Use para responder perguntas sobre documentos técnicos, artigos, relatórios."
            ),
            Tool(
                name="resumo_dataset",
                func=self.resumo_dataset,
                description="Retorna resumo do dataset CSV carregado com dimensões e colunas."
            ),
            Tool(
                name="nomes_colunas",
                func=self.nomes_colunas,
                description="Retorna os nomes de todas as colunas do dataset CSV."
            ),
            Tool(
                name="estatisticas_coluna",
                func=self.estatisticas_coluna,
                description="Calcula estatísticas descritivas (média, desvio, mín, máx) para uma coluna específica do dataset. Argumento: nome da coluna."
            ),
            Tool(
                name="criar_histograma",
                func=self.criar_histograma,
                description="Cria um histograma para visualizar a distribuição de uma coluna numérica. Argumento: nome da coluna."
            ),
            Tool(
                name="criar_boxplot",
                func=self.criar_boxplot,
                description="Cria um boxplot para visualizar outliers e quartis de uma coluna numérica. Argumento: nome da coluna."
            ),
            Tool(
                name="ajuda",
                func=self.ajuda,
                description="Mostra exemplos de perguntas que podem ser feitas ao assistente."
            ),
        ]

        rag_info = "Você tem acesso a documentos PDF através da ferramenta 'buscar_documentos'." if self.tem_rag_disponivel() else "RAG não disponível."

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Você é um assistente de Análise de Dados inteligente e prestativo.

SUAS CAPACIDADES:
1. 📚 Consultar documentos PDF usando 'buscar_documentos'
2. 📊 Analisar datasets CSV usando ferramentas específicas
3. 📈 Criar visualizações de dados
4. 💡 Responder perguntas conceituais diretamente

{rag_info}

REGRAS CRÍTICAS:

1. PERGUNTAS SOBRE DADOS DO DATASET:
   - "qual a média das colunas" → USE nomes_colunas PRIMEIRO, depois estatisticas_coluna
   - "quais as colunas" → USE nomes_colunas
   - "estatísticas da coluna X" → USE estatisticas_coluna
   - "resumo do dataset" → USE resumo_dataset
   - NUNCA invente valores de dados. SEMPRE use as ferramentas!

2. PERGUNTAS CONCEITUAIS:
   - "o que é média?" → Responda diretamente
   - "diferença entre média e mediana?" → Responda diretamente
   - Não precisa de ferramentas para explicar conceitos

3. WORKFLOW CORRETO:
   Para "média das colunas do dataset":
   a) Primeiro: nomes_colunas() para ver quais colunas existem
   b) Depois: estatisticas_coluna() para cada coluna numérica
   c) Finalmente: apresente os resultados reais obtidos

IMPORTANTE: Sempre que a pergunta mencionar "dataset" ou "dados", você DEVE usar as ferramentas!"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agente = create_tool_calling_agent(self.llm, ferramentas, prompt)
        memoria = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )

        executor = AgentExecutor(
            agent=agente,
            tools=ferramentas,
            memory=memoria,
            verbose=True,  # Ativar para debug
            handle_parsing_errors=True,
            max_iterations=10,  # Aumentar iterações
            early_stopping_method="generate",
            return_intermediate_steps=False
        )

        print("✅ Agente configurado com sucesso!")
        return executor


# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal"""
    print("\n" + "=" * 60)
    print("🤖 ASSISTENTE INTELIGENTE - RAG + ANÁLISE DE DADOS")
    print("=" * 60)

    try:
        agente = AgenteEDAComRAG()
        executor = agente.criar_executor()

        print("\n" + "=" * 60)
        print("✅ Sistema pronto!")
        print(f"📚 RAG: {'Ativo' if agente.tem_rag_disponivel() else 'Desativado'}")
        print(f"📊 CSV: {'Carregado' if agente.tem_dataset_carregado() else 'Não carregado'}")
        print("\n💡 Digite 'ajuda' para exemplos | 'sair' para terminar")
        print("=" * 60)

        while True:
            try:
                pergunta = input("\n🎯 Pergunta: ").strip()

                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                elif pergunta.lower() == 'limpar':
                    executor.memory.clear()
                    print("🧹 Memória limpa!")
                    continue
                elif not pergunta:
                    continue

                resposta = executor.invoke({"input": pergunta})
                print(f"\n🤖 {resposta['output']}")

            except KeyboardInterrupt:
                print("\n👋 Interrompido.")
                break
            except Exception as e:
                print(f"❌ Erro: {str(e)[:100]}...")

    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        return


if __name__ == "__main__":
    main()