# vectorstore_creator.py - Vector database creator for RAG (ENGLISH VERSION)

import sys
from pathlib import Path

# Imports with fallbacks for compatibility
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
except ImportError:
    print("❌ Error: langchain_community not found!")
    print("💡 Install with: pip install langchain-community")
    sys.exit(1)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("❌ Error: langchain_text_splitters not found!")
    print("💡 Install with: pip install langchain-text-splitters")
    sys.exit(1)

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("❌ Error: Chroma not found in langchain_community!")
    print("💡 Install with: pip install langchain-community chromadb")
    sys.exit(1)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("❌ Error: HuggingFaceEmbeddings not found!")
    print("💡 Install with: pip install langchain-community sentence-transformers")
    sys.exit(1)


class VectorStoreCreator:
    def __init__(self, rag_dir: str = "rag_files", vectorstore_dir: str = "vectorstore"):
        self.rag_dir = Path(rag_dir)
        self.vectorstore_dir = Path(vectorstore_dir)

    def create_vectorstore(self) -> bool:
        """Creates the vector database from PDF documents"""
        print("🚀 Starting vector database creation...")

        if not self._validate_input():
            return False

        try:
            # 1. Load documents
            documents = self._load_documents()
            if not documents:
                return False

            # 2. Split into chunks
            chunks = self._split_documents(documents)

            # 3. Create embeddings and vectorstore
            self._create_vectorstore(chunks)

            # 4. Save processing stats
            self._save_stats(documents, chunks)

            print("✅ Vector database successfully created!")
            return True

        except Exception as e:
            print(f"❌ Error creating vector database: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _validate_input(self) -> bool:
        """Validates input folder and checks for PDFs"""
        if not self.rag_dir.exists():
            print(f"❌ Folder '{self.rag_dir}' not found!")
            print("💡 Create the folder and add your PDF files.")
            return False

        pdf_files = list(self.rag_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in '{self.rag_dir}'")
            return False

        print(f"📚 Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")

        return True

    def _load_documents(self):
        """Loads PDF documents"""
        print("📂 Loading documents...")
        try:
            loader = PyPDFDirectoryLoader(str(self.rag_dir))
            documents = loader.load()
            print(f"✅ {len(documents)} pages loaded")
            return documents
        except Exception as e:
            print(f"❌ Error loading PDFs: {e}")
            print("💡 Make sure you installed: pip install pypdf")
            return None

    def _split_documents(self, documents):
        """Splits documents into chunks"""
        print("✂️  Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"📄 {len(chunks)} chunks created")
        return chunks

    def _create_vectorstore(self, chunks):
        """Creates and persists the vector database"""
        print("🔨 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("💾 Saving vector database...")
        if self.vectorstore_dir.exists():
            print("⚠️  Existing directory will be overwritten")

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.vectorstore_dir)
        )

        print(f"🗄️ Vectorstore saved at: {self.vectorstore_dir}")

    def _save_stats(self, documents, chunks):
        """Saves processing statistics"""
        stats = {
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "rag_directory": str(self.rag_dir),
            "vectorstore_directory": str(self.vectorstore_dir),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }

        stats_file = self.vectorstore_dir / "processing_stats.json"
        import json
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print("📊 Processing stats saved:")
        print(f"   - Pages processed: {stats['total_pages']}")
        print(f"   - Chunks created: {stats['total_chunks']}")
        print(f"   - Embedding model: {stats['embedding_model']}")


def load_vectorstore(vectorstore_dir: str = "vectorstore"):
    """Loads an existing vectorstore from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings
    )

    return vectorstore


def main():
    """Main entry point"""
    print("🎯 Vector Database Creator")
    print("=" * 50)

    creator = VectorStoreCreator(
        rag_dir="rag_files",
        vectorstore_dir="vectorstore"
    )

    success = creator.create_vectorstore()

    if success:
        print("\n✅ Process completed!")
        print("📁 Vector database is ready at: ./vectorstore/")
        print("💡 You can now use it with your agent.")
    else:
        print("\n❌ Failed to create vector database")
        sys.exit(1)


if __name__ == "__main__":
    main()
