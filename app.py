# app.py - Interface Streamlit para o Agente com RAG (VERSÃO CORRIGIDA FINAL)

import streamlit as st
import os
import sys
from pathlib import Path
import base64
import re

# Adicionar o diretório atual ao path para importar o agente
# Isso garante que o 'import agent' funcione corretamente
sys.path.append(str(Path(__file__).parent))
from agent import AgenteEDAComRAG


# <<< ALTERAÇÃO PRINCIPAL: FUNÇÃO DE CACHE MOVIDA PARA FORA DA CLASSE >>>
# Esta função agora é independente e pode ser cacheada pelo Streamlit,
# pois não recebe o argumento 'self'.
@st.cache_resource
def carregar_agente_e_executor():
    """
    Inicializa o agente e o executor uma única vez e armazena em cache.
    Esta é uma operação cara, por isso o cache é essencial.
    """
    try:
        print("--- INICIALIZANDO AGENTE PELA PRIMEIRA VEZ ---")
        agente = AgenteEDAComRAG()
        executor = agente.criar_executor()
        return agente, executor
    except Exception as e:
        # Se a inicialização falhar aqui, o erro será exibido na interface
        st.error(f"Erro fatal ao inicializar o agente: {e}")
        # Interrompe a execução do app se o componente principal não puder ser carregado
        st.stop()


class StreamlitAgenteInterface:
    """Interface Streamlit para o Agente com RAG"""

    def __init__(self):
        """
        Construtor da interface. Apenas chama a função cacheada para obter
        o agente e o executor, garantindo que a inicialização ocorra apenas uma vez.
        """
        # Chama a função global cacheada para obter as instâncias do agente e executor
        self.agente, self.executor = carregar_agente_e_executor()

    def limpar_caracteres(self, texto: str) -> str:
        """Remove caracteres de controle problemáticos, mas preserva UTF-8."""
        if not isinstance(texto, str):
            return ""
        # Remove caracteres que podem quebrar a renderização, mantendo acentos, etc.
        return texto.replace('\x00', '').replace('\ufffd', '')

    def processar_pergunta(self, pergunta: str) -> str:
        """Processa uma pergunta através do executor do agente."""
        if not self.executor:
            st.error("O executor do agente não está disponível.")
            return "Erro: Agente não inicializado."

        with st.spinner("O agente está pensando..."):
            try:
                # O agente agora deve formular uma resposta final em linguagem natural
                resposta = self.executor.invoke({"input": pergunta})
                return self.limpar_caracteres(resposta['output'])
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
                return "Desculpe, não consegui processar sua solicitação no momento."

    def formatar_e_exibir_resposta(self, resposta_texto: str):
        """Formata e exibe a resposta, tratando texto e imagens de gráficos."""
        if not resposta_texto or not resposta_texto.strip():
            return

        # Separa o texto das tags de imagem para exibi-los corretamente
        partes = re.split(r'(!\[Gráfico\]\(data:image/png;base64,[^)]+\))', resposta_texto)

        for parte in partes:
            # Se a parte for uma tag de imagem
            if parte.startswith("![Gráfico]"):
                img_base64 = re.search(r'base64,([^)]+)', parte)
                if img_base64:
                    try:
                        # Decodifica e exibe a imagem
                        st.image(base64.b64decode(img_base64.group(1)), use_column_width=True)
                    except Exception as e:
                        st.error(f"Erro ao exibir o gráfico: {e}")
            # Se for texto, exibe com suporte a Markdown
            elif parte.strip():
                st.markdown(parte)

    def executar(self):
        """Executa a interface principal do Streamlit."""
        st.set_page_config(page_title="Agente IA - Análise de Dados", page_icon="🤖", layout="wide")

        # --- SIDEBAR ---
        with st.sidebar:
            st.title("Painel de Controle")
            st.markdown("---")

            st.header("Status do Sistema")
            st.success("Agente inicializado com sucesso!")
            rag_status = "Ativo" if self.agente.tem_rag_disponivel() else "Inativo"
            csv_status = "Carregado" if self.agente.tem_dataset_carregado() else "Não Carregado"
            st.info(f"**Busca (RAG):** {rag_status}")
            st.info(f"**Dataset CSV:** {csv_status}")

            if self.agente.tem_dataset_carregado():
                st.write(f"**Arquivo:** `{os.path.basename(self.agente.arquivo_carregado)}`")
                st.write(f"**Dimensões:** {self.agente.df.shape[0]:,} linhas × {self.agente.df.shape[1]} colunas")

            st.markdown("---")

            # Botão para limpar o histórico da conversa na tela
            if st.button("Limpar Histórico da Conversa"):
                st.session_state.messages = []
                # Limpa também a memória do agente para um recomeço real
                if self.executor and hasattr(self.executor, 'memory'):
                    self.executor.memory.clear()
                st.rerun()

            # <<< ALTERAÇÃO: Botão para limpar o cache do agente >>>
            if st.button("Reinicializar Agente"):
                # Chama o método .clear() da função cacheada para forçar a recriação
                carregar_agente_e_executor.clear()
                st.session_state.clear()
                st.rerun()

        # --- ÁREA PRINCIPAL DO CHAT ---
        st.title("🤖 Agente IA para Análise de Dados")
        st.caption("Faça perguntas sobre seus documentos (RAG) ou analise arquivos CSV.")

        # Inicializa o histórico de chat se não existir
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Exibe as mensagens do histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:  # role == "assistant"
                    self.formatar_e_exibir_resposta(message["content"])

        # Captura o input do usuário
        if prompt := st.chat_input("Ex: o que é desvio padrão? ou resumo do dataset"):
            # Adiciona a pergunta do usuário ao histórico e exibe
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Processa a pergunta e exibe a resposta do agente
            with st.chat_message("assistant"):
                resposta_agente = self.processar_pergunta(prompt)
                self.formatar_e_exibir_resposta(resposta_agente)
                # Adiciona a resposta do agente ao histórico
                st.session_state.messages.append({"role": "assistant", "content": resposta_agente})


def main():
    """Função principal para iniciar a aplicação."""
    interface = StreamlitAgenteInterface()
    interface.executar()


if __name__ == "__main__":
    main()