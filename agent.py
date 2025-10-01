# agent.py - Agente com RAG integrado (CORRIGIDO)

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

        # Usar modelo mais leve para evitar erro 413
        model_name = "llama-3.1-8b-instant"

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name=model_name,
            max_tokens=1000  # Limitar tokens de resposta
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
                search_kwargs={"k": 2}  # Reduzir para 2 resultados
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
        """
        if not self.tem_rag_disponivel():
            return "❌ RAG não está disponível."

        if not pergunta:
            return "❌ Forneça uma pergunta."

        try:
            # Buscar documentos relevantes
            docs = self.retriever.get_relevant_documents(pergunta)

            if not docs:
                return "📭 Nenhum documento relevante encontrado."

            # Formatar resultados de forma mais concisa
            resultado = f"📚 {len(docs)} trechos relevantes:\n\n"

            for i, doc in enumerate(docs, 1):
                conteudo = doc.page_content[:200]  # Reduzir tamanho
                fonte = doc.metadata.get('source', 'Desconhecida')
                pagina = doc.metadata.get('page', '?')

                resultado += f"🔹 {i} ({Path(fonte).name}, Pág {pagina}):\n"
                resultado += f"{conteudo}...\n\n"

            return resultado

        except Exception as e:
            return f"❌ Erro na busca: {str(e)}"

    def resumo_dataset(self, query: str = "") -> str:
        """Retorna resumo do dataset CSV carregado."""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset CSV carregado."

        nome = os.path.basename(self.arquivo_carregado)
        return (f"📊 Dataset: {nome}\n"
                f"📏 Dimensões: {self.df.shape[0]:,} linhas × {self.df.shape[1]} colunas\n"
                f"📋 Colunas: {', '.join(self.df.columns[:5])}...")

    def nomes_colunas(self, query: str = "") -> str:
        """Retorna nomes das colunas do dataset."""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."

        colunas = list(self.df.columns)
        if len(colunas) > 10:
            return f"📋 {len(colunas)} colunas: {', '.join(colunas[:8])}..."
        return f"📋 Colunas: {', '.join(colunas)}"

    def estatisticas_coluna(self, coluna: str) -> str:
        """Calcula estatísticas descritivas para uma coluna"""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."
        if coluna not in self.df.columns:
            return f"❌ Coluna '{coluna}' não encontrada."

        try:
            if pd.api.types.is_numeric_dtype(self.df[coluna]):
                stats = self.df[coluna].describe()
                return (f"📊 '{coluna}':\n"
                        f"  • Média: {stats['mean']:.2f}\n"
                        f"  • Desvio: {stats['std']:.2f}\n"
                        f"  • Min/Max: {stats['min']:.2f}/{stats['max']:.2f}\n"
                        f"  • Mediana: {stats['50%']:.2f}")
            else:
                unique = self.df[coluna].nunique()
                top = self.df[coluna].value_counts().head(2)  # Reduzir para top 2
                return (f"📝 '{coluna}':\n"
                        f"  • Únicos: {unique}\n"
                        f"  • Top 2:\n{top.to_string()}")
        except Exception as e:
            return f"❌ Erro ao calcular estatísticas: {str(e)}"

    def criar_histograma(self, coluna: str) -> str:
        """Cria um histograma para uma coluna numérica"""
        return self.criar_grafico("histograma", coluna)

    def criar_boxplot(self, coluna: str) -> str:
        """Cria um boxplot para uma coluna numérica"""
        return self.criar_grafico("boxplot", coluna)

    def criar_grafico(self, tipo: str, coluna: str) -> str:
        """Cria gráfico (histograma ou boxplot) para coluna numérica"""
        if not self.tem_dataset_carregado():
            return "❌ Nenhum dataset carregado."
        if coluna not in self.df.columns:
            return f"❌ Coluna '{coluna}' não encontrada."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return f"❌ '{coluna}' não é numérica."

        plt.figure(figsize=(8, 5))  # Reduzir tamanho

        try:
            if tipo.lower() == "histograma":
                plt.hist(self.df[coluna].dropna(), bins=20, alpha=0.7, edgecolor='black')  # Reduzir bins
                plt.title(f'Histograma - {coluna}')
                plt.xlabel(coluna)
                plt.ylabel('Frequência')
            elif tipo.lower() == "boxplot":
                plt.boxplot(self.df[coluna].dropna())
                plt.title(f'Boxplot - {coluna}')
                plt.ylabel(coluna)
            else:
                return "❌ Tipo inválido."

            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=80)  # Reduzir DPI
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"📈 Gráfico criado!\n![Gráfico](data:image/png;base64,{img_base64})"

        except Exception as e:
            plt.close()
            return f"❌ Erro: {str(e)}"

    def ajuda(self, query: str = "") -> str:
        """Mostra exemplos de perguntas."""
        rag_status = "✅ Ativo" if self.tem_rag_disponivel() else "❌ Desativado"
        csv_status = "✅ Carregado" if self.tem_dataset_carregado() else "❌ Não carregado"

        return f"""
💡 EXEMPLOS:

📚 PDFs (RAG {rag_status}):
- "o que é machine learning?"
- "explique desvio padrão"

📊 CSV ({csv_status}):
- "resumo do dataset"
- "nomes das colunas"
- "estatísticas da coluna V1"
- "histograma de Amount"
"""

    # ==================== CRIAÇÃO DO EXECUTOR ====================

    def criar_executor(self) -> AgentExecutor:
        """Cria o AgentExecutor com todas as ferramentas"""

        # Criar ferramentas
        ferramentas = [
            Tool(
                name="buscar_documentos",
                func=self.buscar_documentos,
                description="Busca em documentos PDF. Use para perguntas sobre conceitos técnicos."
            ),
            Tool(
                name="resumo_dataset",
                func=self.resumo_dataset,
                description="Resumo do dataset CSV com dimensões e colunas."
            ),
            Tool(
                name="nomes_colunas",
                func=self.nomes_colunas,
                description="Lista os nomes das colunas do dataset."
            ),
            Tool(
                name="estatisticas_coluna",
                func=self.estatisticas_coluna,
                description="Estatísticas de uma coluna específica. Argumento: nome_da_coluna"
            ),
            Tool(
                name="criar_histograma",
                func=self.criar_histograma,
                description="Histograma para coluna numérica. Argumento: nome_da_coluna"
            ),
            Tool(
                name="criar_boxplot",
                func=self.criar_boxplot,
                description="Boxplot para coluna numérica. Argumento: nome_da_coluna"
            ),
            Tool(
                name="ajuda",
                func=self.ajuda,
                description="Exemplos de perguntas."
            ),
        ]

        # Prompt mais curto e direto
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""Você é um assistente de análise de dados.

CAPACIDADES:
- Consultar PDFs com 'buscar_documentos'
- Analisar CSV com as ferramentas
- Criar visualizações

REGRAS:
1. Para dados do dataset: SEMPRE use as ferramentas
2. Para conceitos gerais: responda diretamente
3. Seja conciso nas respostas

RAG: {'✅ Ativo' if self.tem_rag_disponivel() else '❌ Inativo'}
CSV: {'✅ Carregado' if self.tem_dataset_carregado() else '❌ Não carregado'}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agente = create_tool_calling_agent(self.llm, ferramentas, prompt)

        memoria = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            max_token_limit=1000  # Limitar memória
        )

        executor = AgentExecutor(
            agent=agente,
            tools=ferramentas,
            memory=memoria,
            verbose=False,  # Desativar verbose para reduzir output
            handle_parsing_errors=True,
            max_iterations=3,  # Reduzir iterações
            early_stopping_method="generate"
        )

        print("✅ Agente configurado com sucesso!")
        return executor


# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal"""
    print("\n" + "=" * 60)
    print("🤖 ASSISTENTE - RAG + ANÁLISE DE DADOS")
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