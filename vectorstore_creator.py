# vectorstore_creator.py

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings



class VectorStoreCreator:
    def __init__(self, rag_dir: str = "rag_files", vectorstore_dir: str = "vectorstore"):
        self.rag_dir = Path(rag_dir)
        self.vectorstore_dir = Path(vectorstore_dir)

    def create_vectorstore(self) -> bool:
        """Cria o banco de dados vetorial a partir dos PDFs"""
        print("🚀 Iniciando criação do banco de dados vetorial...")

        # Verificações iniciais
        if not self._validate_input():
            return False

        try:
            # 1. Carregar documentos
            documents = self._load_documents()
            if not documents:
                return False

            # 2. Dividir em chunks
            chunks = self._split_documents(documents)

            # 3. Criar embeddings e vectorstore
            vectorstore = self._create_vectorstore(chunks)

            # 4. Salvar estatísticas
            self._save_stats(documents, chunks)

            print("✅ Banco de dados vetorial criado com sucesso!")
            return True

        except Exception as e:
            print(f"❌ Erro ao criar banco de dados: {e}")
            return False

    def _validate_input(self) -> bool:
        """Valida se a pasta de entrada existe e tem PDFs"""
        if not self.rag_dir.exists():
            print(f"❌ Pasta '{self.rag_dir}' não encontrada!")
            print(f"💡 Crie a pasta e adicione seus PDFs")
            return False

        pdf_files = list(self.rag_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ Nenhum arquivo PDF encontrado em '{self.rag_dir}'")
            return False

        print(f"📚 Encontrados {len(pdf_files)} arquivos PDF:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")

        return True

    def _load_documents(self):
        """Carrega documentos PDF"""
        print("📂 Carregando documentos...")
        try:
            loader = PyPDFDirectoryLoader(str(self.rag_dir))
            documents = loader.load()
            print(f"✅ {len(documents)} páginas carregadas")
            return documents
        except Exception as e:
            print(f"❌ Erro ao carregar PDFs: {e}")
            return None

    def _split_documents(self, documents):
        """Divide documentos em chunks"""
        print("✂️  Dividindo documentos em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 {len(chunks)} chunks criados")
        return chunks

    def _create_vectorstore(self, chunks):
        """Cria o banco de dados vetorial"""
        print("🔨 Criando embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("💾 Salvando banco vetorial...")
        # Limpar diretório existente se necessário
        if self.vectorstore_dir.exists():
            print("⚠️  Diretório existente será sobrescrito")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.vectorstore_dir)
        )

        print(f"🗄️ Vectorstore salvo em: {self.vectorstore_dir}")
        return vectorstore

    def _save_stats(self, documents, chunks):
        """Salva estatísticas do processamento"""
        stats = {
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "rag_directory": str(self.rag_dir),
            "vectorstore_directory": str(self.vectorstore_dir),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }

        # Salvar estatísticas em arquivo
        stats_file = self.vectorstore_dir / "processing_stats.json"
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print("📊 Estatísticas salvas:")
        print(f"   - Páginas processadas: {stats['total_pages']}")
        print(f"   - Chunks criados: {stats['total_chunks']}")
        print(f"   - Modelo de embeddings: {stats['embedding_model']}")


def main():
    """Função principal"""
    print("🎯 Criador de Banco de Dados Vetorial")
    print("=" * 50)

    creator = VectorStoreCreator(
        rag_dir="rag_files",
        vectorstore_dir="vectorstore"
    )

    success = creator.create_vectorstore()

    if success:
        print("\n✅ Processo concluído!")
        print("📍 O banco de dados está pronto em: ./vectorstore/")
        print("💡 Agora você pode usar com seu agente:")
        print("""
from vectorstore_creator import load_vectorstore
from langchain.embeddings import HuggingFaceEmbeddings

# Carregar o vectorstore existente
vectorstore = load_vectorstore("vectorstore")
retriever = vectorstore.as_retriever()
        """)
    else:
        print("\n❌ Falha na criação do banco de dados")
        sys.exit(1)


def load_vectorstore(vectorstore_dir: str = "vectorstore"):
    """Função para carregar o vectorstore existente"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings
    )

    return vectorstore


if __name__ == "__main__":
    main()