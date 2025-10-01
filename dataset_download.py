import kagglehub
import os
from pathlib import Path
import shutil

def download_creditcard_fraud_dataset():
    """Baixa o dataset de fraudes em cartão de crédito para a pasta data"""
    
    # Criar pasta data se não existir
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Pasta 'data' criada/verificada em: {data_dir.absolute()}")
    
    try:
        # Download do dataset
        print("⬇️  Fazendo download do dataset creditcardfraud...")
        download_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"✅ Dataset baixado em: {download_path}")
        
        # Procurar pelo arquivo creditcard.csv (arquivo principal do dataset)
        source_csv = Path(download_path) / "creditcard.csv"
        
        if source_csv.exists():
            # Caminho de destino
            destination_csv = data_dir / "creditcard.csv"
            
            # Copiar arquivo
            shutil.copy2(source_csv, destination_csv)
            print(f"✅ Arquivo copiado: creditcard.csv -> {destination_csv}")
            
            # Verificar informações do arquivo
            file_size = destination_csv.stat().st_size / (1024 * 1024)  # MB
            print(f"📊 Tamanho do arquivo: {file_size:.2f} MB")
            
            return True
        else:
            # Listar todos os arquivos disponíveis se o principal não for encontrado
            print("❌ Arquivo creditcard.csv não encontrado. Arquivos disponíveis:")
            for file_path in Path(download_path).glob("*"):
                print(f"   - {file_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao baixar ou copiar o dataset: {e}")
        return False

if __name__ == "__main__":
    success = download_creditcard_fraud_dataset()
    if success:
        print("\n🎉 Dataset de fraudes em cartão de crédito baixado com sucesso!")
        print("📍 Localização: ./data/creditcard.csv")
    else:
        print("\n💥 Falha no download do dataset.")