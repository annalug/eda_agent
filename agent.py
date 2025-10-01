# agent.py - Agente com RAG integrado (VERSÃO FINAL COM MODELO CORRETO)

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
from langchain_core.messages import SystemMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic.v1 import BaseModel, Field


# --- Schemas de Argumentos para as Ferramentas ---
class BuscarDocumentosArgs(BaseModel):
    pergunta: str = Field(description="A pergunta ou conceito a ser buscado nos documentos PDF.")

class SemArgumentosArgs(BaseModel):
    """Schema vazio para ferramentas que não precisam de argumentos"""
    pass

class EstatisticasColunaArgs(BaseModel):
    coluna: str = Field(description="O nome exato da coluna para calcular as estatísticas.")


class GraficoArgs(BaseModel):
    coluna: str = Field(description="O nome exato da coluna numérica para gerar o gráfico.")


class AgenteEDAComRAG:
    def __init__(self):
        self.llm = None
        self.df = None
        self.arquivo_carregado = None
        self.vectorstore = None
        self.retriever = None
        print("🚀 Inicializando Agente...")
        self._carregar_configuracoes()
        self._carregar_vectorstore()
        self._carregar_arquivos_automaticamente()

    def _carregar_configuracoes(self):
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key: raise ValueError("GROQ_API_KEY não encontrada.")

        # <<< ALTERAÇÃO FINAL: USO DO MODELO CORRETO E ATUAL DA GROQ API >>>
        model_name = "llama-3.1-8b-instant"  # ou outro modelo disponível


        self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model_name)
        print(f"✅ LLM configurado: {model_name}")

    def _carregar_vectorstore(self):
        vectorstore_dir = "vectorstore"
        if not Path(vectorstore_dir).exists():
            print("⚠️  RAG desabilitado (pasta 'vectorstore' não encontrada).")
            return
        try:
            print("📚 Carregando banco vetorial...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            if self.vectorstore._collection.count() > 0:
                print("✅ RAG ativado!")
            else:
                self.vectorstore = self.retriever = None; print("⚠️  Vectorstore vazio.")
        except Exception as e:
            self.vectorstore = self.retriever = None;
            print(f"⚠️  Erro ao carregar vectorstore: {e}")

    def _carregar_arquivos_automaticamente(self):
        pasta = "data"
        if not os.path.exists(pasta): os.makedirs(pasta, exist_ok=True)
        arquivos_disponiveis = glob.glob(os.path.join(pasta, "*.csv"))
        if arquivos_disponiveis:
            self._carregar_primeiro_arquivo(arquivos_disponiveis[0])
        else:
            print(f"📁 Nenhum CSV encontrado em '{pasta}'")

    def _carregar_primeiro_arquivo(self, caminho_arquivo):
        nome = os.path.basename(caminho_arquivo)
        print(f"⏳ Carregando '{nome}'...")
        try:
            self.df = pd.read_csv(caminho_arquivo)
            self.arquivo_carregado = caminho_arquivo
            print(f"✅ Arquivo carregado: {self.df.shape[0]:,} linhas × {self.df.shape[1]} colunas")
            if len(self.df) > 10000:
                print("📊 Usando amostra de 10.000 linhas.")
                self.df = self.df.sample(n=10000, random_state=42)
        except Exception as e:
            self.df = None;
            print(f"❌ Erro ao carregar CSV: {e}")

    def tem_dataset_carregado(self):
        return self.df is not None and not self.df.empty

    def tem_rag_disponivel(self):
        return self.retriever is not None

    def buscar_documentos(self, pergunta: str) -> str:
        if not self.tem_rag_disponivel(): return "Sistema de busca (RAG) não está ativo."
        docs = self.retriever.get_relevant_documents(pergunta)
        if not docs: return "Nenhuma informação sobre este tópico foi encontrada nos documentos."
        contexto = "\n\n".join([f"> {doc.page_content}" for doc in docs])
        return f"Com base nos documentos, encontrei o seguinte:\n{contexto}"

    def resumo_dataset(self) -> str:
        if not self.tem_dataset_carregado(): return "Nenhum arquivo CSV está carregado."
        nome = os.path.basename(self.arquivo_carregado)
        return f"O dataset é '{nome}', com {self.df.shape[0]:,} linhas e {self.df.shape[1]} colunas."

    def nomes_colunas(self) -> str:
        if not self.tem_dataset_carregado(): return "Nenhum arquivo CSV está carregado."
        return f"As colunas são: {', '.join(list(self.df.columns))}"

    def estatisticas_gerais(self) -> str:
        if not self.tem_dataset_carregado(): return "Nenhum arquivo CSV está carregado."
        df_numeric = self.df.select_dtypes(include=np.number)
        if df_numeric.empty: return "O dataset não possui colunas numéricas."
        stats = df_numeric.describe().round(2)
        return f"Aqui estão as estatísticas para as colunas numéricas:\n\n```\n{stats.to_string()}\n```"

    def estatisticas_coluna(self, coluna: str) -> str:
        if not self.tem_dataset_carregado(): return "Nenhum arquivo CSV está carregado."
        if coluna not in self.df.columns: return f"A coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]): return f"A coluna '{coluna}' não é numérica."
        stats = self.df[coluna].describe()
        return f"Estatísticas da coluna '{coluna}': Média={stats['mean']:.2f}, Desvio Padrão={stats['std']:.2f}, Mín={stats['min']:.2f}, Máx={stats['max']:.2f}."

    def criar_grafico(self, tipo: str, coluna: str) -> str:
        if not self.tem_dataset_carregado(): return "Nenhum arquivo CSV está carregado."
        if coluna not in self.df.columns: return f"A coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]): return f"A coluna '{coluna}' não é numérica."
        plt.style.use('seaborn-v0_8-darkgrid');
        plt.figure(figsize=(8, 4))
        try:
            if tipo == "histograma":
                plt.hist(self.df[coluna].dropna(), bins=30, edgecolor='black'); plt.title(f'Histograma - {coluna}')
            elif tipo == "boxplot":
                plt.boxplot(self.df[coluna].dropna(), vert=False); plt.title(f'Boxplot - {coluna}')
            buffer = io.BytesIO();
            plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight');
            plt.close()
            img_base_64 = base64.b64encode(buffer.getvalue()).decode()
            return f"![Gráfico](data:image/png;base64,{img_base_64})"
        except Exception as e:
            plt.close();
            return f"Erro ao gerar gráfico: {e}"

    def criar_histograma(self, coluna: str) -> str:
        return self.criar_grafico("histograma", coluna)

    def criar_boxplot(self, coluna: str) -> str:
        return self.criar_grafico("boxplot", coluna)

    def criar_executor(self) -> AgentExecutor:
        ferramentas = [
            Tool(name="buscar_documentos", func=self.buscar_documentos,
                 description="Use para perguntas sobre conceitos, definições ou explicações (ex: 'o que é média?').",
                 args_schema=BuscarDocumentosArgs),
            Tool(name="estatisticas_gerais", func=self.estatisticas_gerais,
                 description="Use para perguntas sobre estatísticas de VÁRIAS colunas ao mesmo tempo (ex: 'quais as médias das colunas?', 'estatísticas gerais')."),
            Tool(name="resumo_dataset", func=self.resumo_dataset,
                 description="Use para perguntas sobre o nome do arquivo, número de linhas ou colunas."),
            Tool(name="nomes_colunas",func=self.nomes_colunas,
     description="Use para listar os nomes de TODAS as colunas.",
     args_schema=SemArgumentosArgs),
            Tool(name="estatisticas_coluna", func=self.estatisticas_coluna,
                 description="Use para obter estatísticas de UMA ÚNICA coluna.", args_schema=EstatisticasColunaArgs),
            Tool(name="criar_histograma", func=self.criar_histograma,
                 description="Use para criar um histograma de UMA ÚNICA coluna.", args_schema=GraficoArgs),
            Tool(name="criar_boxplot", func=self.criar_boxplot,
                 description="Use para criar um boxplot de UMA ÚNICA coluna.", args_schema=GraficoArgs),
        ]

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Você é um assistente de análise de dados prestativo.
- Primeiro, pense sobre qual ferramenta usar para responder à pergunta do usuário.
- Em seguida, use a ferramenta escolhida.
- Finalmente, use o resultado da ferramenta para dar uma resposta final e completa ao usuário em português."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agente = create_tool_calling_agent(self.llm, ferramentas, prompt)
        memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")
        executor = AgentExecutor(
            agent=agente,
            tools=ferramentas,
            memory=memoria,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        print("✅ Agente configurado com sucesso!")
        return executor


def main():
    agente = AgenteEDAComRAG()
    executor = agente.criar_executor()
    print("=" * 60 + "\n✅ Sistema pronto!\n" + "=" * 60)
    while True:
        pergunta = input("\n🎯 Pergunta: ").strip()
        if pergunta.lower() in ['sair', 'exit', 'quit']: print("👋"); break
        if not pergunta: continue
        try:
            resposta = executor.invoke({"input": pergunta})
            print(f"\n🤖 {resposta['output']}")
        except Exception as e:
            print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()