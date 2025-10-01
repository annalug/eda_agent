# app.py - Interface Streamlit para o Agente com RAG (CORRIGIDO)

import streamlit as st
import os
import sys
from pathlib import Path

# Adicionar o diretório atual ao path para importar o agente
sys.path.append(str(Path(__file__).parent))

from agent import AgenteEDAComRAG


class StreamlitAgenteInterface:
    """Interface Streamlit para o Agente com RAG"""

    def __init__(self):
        self.agente = None
        self.executor = None

    def inicializar_agente(self):
        """Inicializa o agente"""
        try:
            if self.agente is None:
                self.agente = AgenteEDAComRAG()
                self.executor = self.agente.criar_executor()
            return True
        except Exception as e:
            st.error(f"❌ Erro ao inicializar agente: {e}")
            return False

    def processar_pergunta(self, pergunta):
        """Processa uma pergunta através do agente"""
        if not self.executor:
            return "⚠️ Agente não inicializado. Recarregue a página."

        try:
            with st.spinner("🤖 Processando..."):
                resposta = self.executor.invoke({"input": pergunta})
                return resposta['output']
        except Exception as e:
            return f"❌ Erro: {str(e)[:200]}"

    def formatar_resposta(self, resposta):
        """Formata a resposta para exibição no Streamlit"""
        if "![Gráfico]" in resposta:
            # Extrair a imagem base64
            import base64
            from io import BytesIO
            import re

            # Encontrar todas as imagens base64 na resposta
            imagens = re.findall(r'!\[Gráfico\]\(data:image/png;base64,([^)]+)\)', resposta)

            for img_base64 in imagens:
                try:
                    # Decodificar a imagem
                    img_data = base64.b64decode(img_base64)

                    # Exibir a imagem
                    st.image(img_data, caption="Gráfico gerado", use_column_width=True)

                    # Remover o marcador da imagem do texto
                    resposta = resposta.replace(f"![Gráfico](data:image/png;base64,{img_base64})", "")
                except Exception as e:
                    st.error(f"Erro ao exibir gráfico: {e}")

        # Exibir o texto restante - CORREÇÃO: usar st.markdown com HTML para garantir contraste
        if resposta.strip():
            # Substituir quebras de linha por HTML
            resposta_formatada = resposta.replace('\n', '<br>')
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                color: #212529;
                line-height: 1.6;
            ">
                {resposta_formatada}
            </div>
            """, unsafe_allow_html=True)

    def executar(self):
        """Executa a interface Streamlit"""
        st.set_page_config(
            page_title="Agente IA - RAG + Análise de Dados",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # CSS personalizado - CORRIGIDO para melhor contraste
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 1rem 0;
            color: #000000;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            color: #000000;
        }
        .user-message {
            background-color: #e6f3ff;
            border-left: 4px solid #1f77b4;
            color: #000000;
        }
        .assistant-message {
            background-color: #f0f8f0;
            border-left: 4px solid #2ca02c;
            color: #000000;
        }
        /* Garantir que todo texto seja visível */
        .stMarkdown, .stText, .stWrite {
            color: #000000 !important;
        }
        /* Corrigir cores dos expanders */
        .streamlit-expanderHeader {
            color: #000000 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.markdown('<h1 class="main-header">🤖 Agente IA - RAG + Análise de Dados</h1>', unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("📊 Configurações")

            if st.button("🔄 Reinicializar Agente"):
                st.session_state.clear()
                self.agente = None
                self.executor = None
                st.rerun()

            st.markdown("---")
            st.header("ℹ️ Status do Sistema")

            # Inicializar agente
            if self.agente is None:
                if self.inicializar_agente():
                    st.success("✅ Agente inicializado!")
                else:
                    st.error("❌ Falha na inicialização")
                    return

            # Status RAG
            rag_status = "✅ Ativo" if self.agente.tem_rag_disponivel() else "❌ Desativado"
            st.write(f"**RAG:** {rag_status}")

            # Status CSV
            csv_status = "✅ Carregado" if self.agente.tem_dataset_carregado() else "❌ Não carregado"
            st.write(f"**Dataset CSV:** {csv_status}")

            if self.agente.tem_dataset_carregado():
                st.write(f"**Arquivo:** {os.path.basename(self.agente.arquivo_carregado)}")
                st.write(f"**Dimensões:** {self.agente.df.shape[0]:,} linhas × {self.agente.df.shape[1]} colunas")

            st.markdown("---")
            st.header("💡 Exemplos de Perguntas")

            with st.expander("📚 Consulta em PDFs (RAG)"):
                st.markdown("""
                - "O que os documentos dizem sobre machine learning?"
                - "Explique o conceito de desvio padrão baseado nos PDFs"
                - "Resumo sobre análise exploratória de dados"
                """)

            with st.expander("📊 Análise de CSV"):
                st.markdown("""
                - "Resumo do dataset"
                - "Nomes das colunas"
                - "Estatísticas da coluna V1"
                - "Criar histograma de Amount"
                - "Criar boxplot de V2"
                """)

            with st.expander("💡 Conceitos Gerais"):
                st.markdown("""
                - "O que é média?"
                - "Diferença entre média e mediana?"
                - "Para que serve um histograma?"
                """)

        # Área principal do chat
        st.header("💬 Chat com o Agente")

        # Inicializar histórico de conversa
        if "historico" not in st.session_state:
            st.session_state.historico = []

        # Exibir histórico de conversa
        for mensagem in st.session_state.historico:
            if mensagem["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong style="color: #000000;">👤 Você:</strong> 
                    <span style="color: #000000;">{mensagem["content"]}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong style="color: #000000;">🤖 Agente:</strong>
                </div>
                """, unsafe_allow_html=True)
                self.formatar_resposta(mensagem["content"])

        # Input do usuário
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                pergunta = st.text_input(
                    "Digite sua pergunta:",
                    placeholder="Ex: Qual a média das colunas numéricas?",
                    label_visibility="collapsed"
                )
            with col2:
                enviar = st.form_submit_button("Enviar")

            if enviar and pergunta:
                # Adicionar pergunta ao histórico
                st.session_state.historico.append({
                    "role": "user",
                    "content": pergunta
                })

                # Processar pergunta
                resposta = self.processar_pergunta(pergunta)

                # Adicionar resposta ao histórico
                st.session_state.historico.append({
                    "role": "assistant",
                    "content": resposta
                })

                st.rerun()

        # Botões de ação
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Limpar Conversa"):
                st.session_state.historico = []
                if self.executor and hasattr(self.executor, 'memory'):
                    self.executor.memory.clear()
                st.rerun()

        with col2:
            if st.button("📋 Exemplos Rápidos"):
                # Criar um container para os exemplos
                with st.container():
                    st.markdown("**Clique em um exemplo:**")

                    exemplos = [
                        "Resumo do dataset",
                        "Nomes das colunas",
                        "O que é desvio padrão?",
                        "Estatísticas da coluna V1"
                    ]

                    # Criar colunas para os exemplos
                    cols = st.columns(2)
                    for i, exemplo in enumerate(exemplos):
                        with cols[i % 2]:
                            if st.button(exemplo, key=f"ex_{i}", use_container_width=True):
                                st.session_state.historico.append({
                                    "role": "user",
                                    "content": exemplo
                                })
                                resposta = self.processar_pergunta(exemplo)
                                st.session_state.historico.append({
                                    "role": "assistant",
                                    "content": resposta
                                })
                                st.rerun()

        with col3:
            if st.button("📊 Visualizar Dados"):
                if self.agente and self.agente.tem_dataset_carregado():
                    with st.expander("📋 Visualização do Dataset", expanded=True):
                        st.dataframe(self.agente.df.head(100))

                        # Estatísticas rápidas
                        st.subheader("📈 Estatísticas Descritivas")
                        st.dataframe(self.agente.df.describe())


def main():
    """Função principal"""
    interface = StreamlitAgenteInterface()
    interface.executar()


if __name__ == "__main__":
    main()