# agent.py - EDA Agent with integrated RAG (FINAL STABLE VERSION)

import os
import glob
import io
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# =========================
# Tool Schemas
# =========================

class SearchDocumentsArgs(BaseModel):
    query: str = Field(description="Question or concept to search for in the documents.")


class ColumnStatsArgs(BaseModel):
    column: str = Field(description="Exact numeric column name to compute statistics.")


class PlotArgs(BaseModel):
    column: str = Field(description="Exact numeric column name to generate the plot.")


class NoArgs(BaseModel):
    """Empty schema for tools with no arguments."""
    pass


# =========================
# Main Agent Class
# =========================

class EDAAgentWithRAG:
    def __init__(self):
        self.llm = None
        self.df = None
        self.loaded_file = None
        self.vectorstore = None
        self.retriever = None

        print("🚀 Initializing agent...")
        self._load_config()
        self._load_vectorstore()
        self._auto_load_files()

    # -------------------------
    # Setup
    # -------------------------

    def _load_config(self):
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found.")

        model_name = "llama-3.1-8b-instant"
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name=model_name
        )
        print(f"✅ LLM configured: {model_name}")

    def _load_vectorstore(self):
        vectorstore_dir = "vectorstore"
        if not Path(vectorstore_dir).exists():
            print("⚠️  RAG disabled (folder 'vectorstore' not found).")
            return
        try:
            print("📚 Loading vectorstore...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vectorstore = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            if self.vectorstore._collection.count() > 0:
                print("✅ RAG enabled!")
            else:
                self.vectorstore = self.retriever = None
                print("⚠️  Vectorstore is empty.")
        except Exception as e:
            self.vectorstore = self.retriever = None
            print(f"⚠️  Error loading vectorstore: {e}")

    def _auto_load_files(self):
        folder = "data"
        os.makedirs(folder, exist_ok=True)
        files = glob.glob(os.path.join(folder, "*.csv"))
        if files:
            self._load_first_file(files[0])
        else:
            print(f"📁 No CSV files found in '{folder}'")

    def _load_first_file(self, file_path):
        name = os.path.basename(file_path)
        print(f"⏳ Loading '{name}'...")
        try:
            self.df = pd.read_csv(file_path)
            self.loaded_file = file_path
            print(f"✅ File loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")

            if len(self.df) > 10000:
                print("📊 Using a sample of 10,000 rows.")
                self.df = self.df.sample(n=10000, random_state=42)
        except Exception as e:
            self.df = None
            print(f"❌ Error loading CSV: {e}")

    # -------------------------
    # Helpers
    # -------------------------

    def has_dataset(self):
        return self.df is not None and not self.df.empty

    def has_rag(self):
        return self.retriever is not None

    # -------------------------
    # Tool Implementations
    # -------------------------

    def search_documents(self, query: str) -> str:
        if not self.has_rag():
            return "Document search system (RAG) is not active."
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found in the documents."

        context = "\n\n".join([f"> {doc.page_content}" for doc in docs])
        return f"Based on the documents, I found:\n{context}"

    def dataset_summary(self) -> str:
        if not self.has_dataset():
            return "No CSV file is loaded."
        name = os.path.basename(self.loaded_file)
        return f"Dataset '{name}': {self.df.shape[0]} rows, {self.df.shape[1]} columns."

    def column_names(self) -> str:
        if not self.has_dataset():
            return "No CSV file is loaded."
        return "Columns: " + ", ".join(self.df.columns)

    # def general_stats(self) -> str:
    #     if not self.has_dataset():
    #         return "No CSV file is loaded."
    #     return (
    #         "Summary statistics were computed. "
    #         "Ask for a specific column if you want detailed statistics."
    #     )

    def column_stats(self, column: str) -> str:
        if not self.has_dataset():
            return "No CSV file is loaded."
        if column not in self.df.columns:
            return f"Column '{column}' does not exist."
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"Column '{column}' is not numeric."

        stats = self.df[column].describe()
        return (
            f"Statistics for '{column}': "
            f"Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, "
            f"Min={stats['min']:.2f}, Max={stats['max']:.2f}."
        )

    def _create_plot(self, plot_type: str, column: str) -> str:
        if not self.has_dataset():
            return "No CSV file is loaded."
        if column not in self.df.columns:
            return f"Column '{column}' does not exist."
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"Column '{column}' is not numeric."

        plt.figure(figsize=(8, 4))
        try:
            if plot_type == "histogram":
                plt.hist(self.df[column].dropna(), bins=30, edgecolor="black")
                plt.title(f"Histogram - {column}")
            elif plot_type == "boxplot":
                plt.boxplot(self.df[column].dropna(), vert=False)
                plt.title(f"Boxplot - {column}")

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
            plt.close()

            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"![Plot](data:image/png;base64,{img_base64})"
        except Exception as e:
            plt.close()
            return f"Error generating plot: {e}"

    def create_histogram(self, column: str) -> str:
        return self._create_plot("histogram", column)

    def create_boxplot(self, column: str) -> str:
        return self._create_plot("boxplot", column)

    # -------------------------
    # Wrappers (no-args tools)
    # -------------------------

    def _dataset_summary_wrapper(self, *args, **kwargs):
        return self.dataset_summary()

    def _column_names_wrapper(self, *args, **kwargs):
        return self.column_names()

    def _general_stats_wrapper(self, *args, **kwargs):
        return self.general_stats()

    # -------------------------
    # Agent Executor
    # -------------------------

    def create_executor(self) -> AgentExecutor:
        tools = [
            StructuredTool(
                name="search_documents",
                func=self.search_documents,
                description=(
                    "Search the loaded documents (RAG) for specific information. "
                    "Use ONLY when the user explicitly asks about the documents. "
                    "Do NOT use for general definitions."
                ),
                args_schema=SearchDocumentsArgs
            ),
            StructuredTool(
                name="dataset_summary",
                func=self._dataset_summary_wrapper,
                description="Get dataset name, number of rows and columns.",
                args_schema=NoArgs
            ),
            StructuredTool(
                name="column_names",
                func=self._column_names_wrapper,
                description="List all column names in the dataset.",
                args_schema=NoArgs
            ),
            # StructuredTool(
            #     name="general_stats",
            #     func=self._general_stats_wrapper,
            #     description=(
            #         "Compute general statistics for the dataset. "
            #         "Use ONLY when explicitly requested. "
            #         "Do NOT use for conceptual questions."
            #     ),
            #     args_schema=NoArgs
            # ),
            StructuredTool(
                name="column_stats",
                func=self.column_stats,
                description="Get statistics for a single numeric column.",
                args_schema=ColumnStatsArgs
            ),
            StructuredTool(
                name="create_histogram",
                func=self.create_histogram,
                description="Create a histogram for a single numeric column.",
                args_schema=PlotArgs
            ),
            StructuredTool(
                name="create_boxplot",
                func=self.create_boxplot,
                description="Create a boxplot for a single numeric column.",
                args_schema=PlotArgs
            ),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an AI assistant for data analysis. "
             "Answer clearly and concisely. "
             "For general definitions or concepts, answer directly without tools. "
             "When you use a tool and get its result, you MUST immediately produce the final answer. "
             "Do NOT call the same tool more than once for the same question. "
             "Use tools only when strictly necessary to access data or documents."
             ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="force",
        )

        print("✅ Agent successfully configured!")
        return executor


def main():
    agent = EDAAgentWithRAG()
    executor = agent.create_executor()
    print("=" * 60 + "\n✅ System ready!\n" + "=" * 60)

    while True:
        question = input("\n🎯 Question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        if not question:
            continue
        try:
            response = executor.invoke({"input": question})
            print(f"\n🤖 {response['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
