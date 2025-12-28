# vectorstore_loader.py - Load existing vector database (ENGLISH VERSION)

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path


def load_vector_database(path: str) -> Chroma:
    """
    Loads an existing vector database.

    Args:
        path: Path to the vectorstore directory.

    Returns:
        Chroma instance or None if an error occurs.
    """
    try:
        if not Path(path).exists():
            print(f"❌ Vectorstore directory not found: {path}")
            return None

        files = list(Path(path).glob("*"))
        if not files:
            print(f"❌ No vectorstore files found in: {path}")
            return None

        print("🔧 Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"📂 Loading vectorstore from: {path}")
        vectorstore = Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )

        count = vectorstore._collection.count()
        print("✅ Vectorstore successfully loaded!")
        print(f"📊 Documents in database: {count}")

        return vectorstore

    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return None


def test_loading():
    """Quick test for vectorstore loading"""
    print("🧪 Testing vectorstore loading...")

    vectorstore = load_vector_database("vectorstore")

    if vectorstore:
        print("✅ Test successful!")

        results = vectorstore.similarity_search("technology", k=2)
        print(f"📄 Documents found in test search: {len(results)}")
        return True
    else:
        print("❌ Loading test failed")
        return False


if __name__ == "__main__":
    test_loading()
