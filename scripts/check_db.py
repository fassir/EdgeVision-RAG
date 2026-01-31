import pickle
import os

"""
Script de Verificação Rápida do Banco de Dados.

Lê o arquivo Pickle local diretamente para verificar se dados estão sendo salvos,
sem precisar subir a API ou o ChromaDB. Útil para debug.
"""

path = "data/vector_store.pkl"
if os.path.exists(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            count = len(data.get('embeddings', []))
            print(f"✅ SUCESSO: O banco de dados contém {count} vetores.")
            if count == 0:
                 print("⚠️ O banco existe mas está vazio. Talvez o vídeo não tenha detectado pessoas?")
    except Exception as e:
        print(f"❌ Erro ao ler o banco: {e}")
else:
    print(f"❌ Arquivo {path} não encontrado.")
