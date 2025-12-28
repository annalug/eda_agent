# app.py - Streamlit interface for the EDA Agent with RAG (ENGLISH VERSION)

import streamlit as st
import os
import sys
from pathlib import Path
import base64
import re

sys.path.append(str(Path(__file__).parent))
from agent import EDAAgentWithRAG


@st.cache_resource
def load_agent_and_executor():
    try:
        print("--- INITIALIZING AGENT FOR THE FIRST TIME ---")
        agent = EDAAgentWithRAG()
        executor = agent.create_executor()
        return agent, executor
    except Exception as e:
        st.error(f"Fatal error while initializing the agent: {e}")
        st.stop()


class StreamlitAgentUI:
    def __init__(self):
        self.agent, self.executor = load_agent_and_executor()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.replace('\x00', '').replace('\ufffd', '')

    def process_question(self, question: str) -> str:
        if not self.executor:
            st.error("Agent executor is not available.")
            return "Error: Agent not initialized."

        with st.spinner("The agent is thinking..."):
            try:
                response = self.executor.invoke({"input": question})
                return self.clean_text(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return "Sorry, I couldn't process your request."

    def display_response(self, response_text: str):
        if not response_text.strip():
            return

        parts = re.split(r'(!\[Plot\]\(data:image/png;base64,[^)]+\))', response_text)

        for part in parts:
            if part.startswith("![Plot]"):
                img_base64 = re.search(r'base64,([^)]+)', part)
                if img_base64:
                    st.image(base64.b64decode(img_base64.group(1)), use_column_width=True)
            elif part.strip():
                st.markdown(part)

    def run(self):
        st.set_page_config(
            page_title="AI Agent - Data Analysis",
            page_icon="🤖",
            layout="wide"
        )

        # Sidebar
        with st.sidebar:
            st.title("Control Panel")
            st.markdown("---")

            st.header("System Status")
            st.success("Agent initialized successfully!")
            rag_status = "Active" if self.agent.has_rag() else "Inactive"
            csv_status = "Loaded" if self.agent.has_dataset() else "Not Loaded"
            st.info(f"**RAG Search:** {rag_status}")
            st.info(f"**CSV Dataset:** {csv_status}")

            if self.agent.has_dataset():
                st.write(f"**File:** `{os.path.basename(self.agent.loaded_file)}`")
                st.write(
                    f"**Shape:** {self.agent.df.shape[0]:,} rows × {self.agent.df.shape[1]} columns"
                )

            st.markdown("---")

            if st.button("Clear Chat History"):
                st.session_state.messages = []
                if self.executor and hasattr(self.executor, "memory"):
                    self.executor.memory.clear()
                st.rerun()

            if st.button("Reinitialize Agent"):
                load_agent_and_executor.clear()
                st.session_state.clear()
                st.rerun()

        # Main area
        st.title("🤖 AI Agent for Data Analysis")
        st.caption("Ask questions about your documents (RAG) or analyze CSV files.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    self.display_response(message["content"])

        if prompt := st.chat_input("e.g., what is standard deviation? or dataset summary"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer = self.process_question(prompt)
                self.display_response(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    ui = StreamlitAgentUI()
    ui.run()


if __name__ == "__main__":
    main()
