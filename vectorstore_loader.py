from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path


def carrega_banco_de_dados_vetorial(path_documentos: str) -> Chroma:
    """
    Carrega o banco de dados vetorial existente

    Args:
        path_documentos: Caminho para a pasta do vectorstore

    Returns:
        Chroma: Instância do banco de dados vetorial ou None em caso de erro
    """
    try:
        # Verificar se o diretório existe
        if not Path(path_documentos).exists():
            print(f"❌ Diretório do vectorstore não encontrado: {path_documentos}")
            return None

        # Verificar se existem arquivos no diretório
        vectorstore_files = list(Path(path_documentos).glob("*"))
        if not vectorstore_files:
            print(f"❌ Nenhum arquivo do vectorstore encontrado em: {path_documentos}")
            return None

        print(f"🔧 Carregando embeddings...")
        # Usando embeddings do HuggingFace (mesmo usado na criação)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"📂 Carregando vectorstore de: {path_documentos}")
        # Carrega o banco de dados vetorial existente
        vectorstore = Chroma(
            persist_directory=path_documentos,
            embedding_function=embeddings
        )

        # Verificar se carregou corretamente
        collection_count = vectorstore._collection.count()
        print(f"✅ Vectorstore carregado com sucesso!")
        print(f"📊 Documentos no banco: {collection_count}")

        return vectorstore

    except Exception as e:
        print(f"❌ Erro ao carregar o banco de dados vetorial: {e}")
        return None


# teste_carregamento.py
def testa_carregamento():
    """Testa o carregamento do vectorstore"""
    print("🧪 Testando carregamento do vectorstore...")
    
    vectorstore = carrega_banco_de_dados_vetorial("vectorstore")
    
    if vectorstore:
        print("✅ Teste bem-sucedido!")
        
        # Teste rápido de busca
        resultados = vectorstore.similarity_search("tecnologia", k=2)
        print(f"📄 Documentos encontrados no teste: {len(resultados)}")
        
        return True
    else:
        print("❌ Falha no carregamento")
        return False

if __name__ == "__main__":
    testa_carregamento()