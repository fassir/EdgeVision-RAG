import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

class VectorDBWrapper:
    """
    Wrapper Híbrido para Banco de Dados Vetorial.
    
    Esta classe abstrai a complexidade do armazenamento de embeddings.
    Ela tenta usar ChromaDB (padrão de mercado) para persistência robusta.
    Se falhar (ex: incompatibilidade de `hnswlib` com Python muito novo ou Windows ARM),
    ela degrada automaticamente (Fallback) para `sklearn.neighbors` + `pickle` (arquivo local).
    
    Atributos:
        storage_path (str): Caminho para o arquivo .pkl (usado no modo Fallback).
        fallback_mode (bool): Flag que indica se estamos usando ChromaDB (False) ou Pickle (True).
        embeddings (list): Cache em memória dos vetores (modo Fallback).
        metadata (list): Cache em memória dos metadados (modo Fallback).
        collection (chromadb.Collection): Referência para a coleção do ChromaDB (se disponível).
    """

    def __init__(self, collection_name="edge_vision", storage_path="data/vector_store.pkl", reset=False):
        """
        Inicializa o wrapper do banco de dados.

        Args:
            collection_name (str): Nome da coleção (tabela) lógica.
            storage_path (str): Onde salvar o arquivo de backup se usar o modo Fallback.
            reset (bool): Se True, APAGA todos os dados anteriores ao iniciar (Limpeza).
        """
        self.storage_path = storage_path
        self.embeddings = []
        self.metadata = [] # Lista de dicionários: {'timestamp': 12.5, 'source': 'video.mp4'}
        self.collection_name = collection_name
        self.knn = None # Modelo NearestNeighbors (Lazy loading)
        self.fallback_mode = True 
        
        # Tenta importar ChromaDB. Bloco try-except garante robustez em qualquer ambiente.
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path="data/chroma_db")
            
            if reset:
                try:
                    self.client.delete_collection(collection_name)
                    print(f"VectorDB: Collection '{collection_name}' resetada (Apagada).")
                except ValueError:
                    pass # Collection pode não existir ainda, ignora erro.
            
            self.collection = self.client.get_or_create_collection(name=collection_name)
            self.fallback_mode = False
            print("VectorDB: Inicializado ChromaDB com sucesso.")
        except ImportError:
            # Entra aqui se 'pip install chromadb' falhou ou se a lib C++ subjacente não compilou.
            print("VectorDB: ChromaDB não encontrado/compatível. Rodando em modo FALLBACK (sklearn + pickle).")
            
            # Lógica de Reset para modo arquivo local
            if reset and os.path.exists(self.storage_path):
                try:
                    os.remove(self.storage_path)
                    print(f"VectorDB: Arquivo local '{self.storage_path}' deletado (Reset).")
                except OSError as e:
                    print(f"VectorDB: Erro ao deletar arquivo local: {e}")
            
            self._load_local_store()

    def _load_local_store(self):
        """Carrega dados do arquivo pickle para a memória RAM (Apenas modo Fallback)."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings', [])
                self.metadata = data.get('metadata', [])
            print(f"VectorDB: Carregados {len(self.embeddings)} registros do armazenamento local.")
        else:
            print("VectorDB: Inicializado armazenamento local vazio.")

    def _save_local_store(self):
        """Persiste dados da memória RAM para o disco (Apenas modo Fallback)."""
        if self.fallback_mode:
            with open(self.storage_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'metadata': self.metadata
                }, f)
            
            # Re-treina o índice KNN sempre que salvamos, para garantir que buscas futuras considerem dados novos
            if len(self.embeddings) > 0:
                # n_neighbors=5 é um padrão razoável, metric='cosine' é crucial para embeddings (CLIP)
                self.knn = NearestNeighbors(n_neighbors=min(5, len(self.embeddings)), metric='cosine')
                self.knn.fit(self.embeddings)

    def add(self, vector, meta_info, id_str):
        """
        Adiciona um novo embedding ao banco de dados.

        Args:
            vector (numpy.ndarray): O vetor de características (embedding).
            meta_info (dict): Metadados associados (timestamp, arquivo de origem, confiança, etc).
            id_str (str): Identificador único (UUID).
        """
        if self.fallback_mode:
            self.embeddings.append(vector)
            self.metadata.append(meta_info)
            # Salva a cada inserção por segurança neste POC. 
            # Em produção, usaria um buffer para salvar a cada N inserções.
            self._save_local_store()
        else:
            self.collection.add(
                documents=["image_feature"], # Campo obrigatório do Chroma, mas não usamos para busca
                embeddings=[vector.tolist()],
                metadatas=[meta_info],
                ids=[id_str]
            )

    def search(self, query_vector, k=5):
        """
        Busca os vizinhos mais próximos (KNN) para um vetor de consulta.

        Args:
            query_vector (numpy.ndarray): Vetor da query de texto.
            k (int): Quantos resultados retornar (Top K).

        Returns:
            list: Lista de dicionários contendo metadados e distância.
                  Ex: [{'metadata': {...}, 'distance': 0.15}, ...]
        """
        if self.fallback_mode:
            if len(self.embeddings) == 0:
                return []
            
            # Inicialização Lazy do modelo KNN se ainda não existir
            if self.knn is None:
                self.knn = NearestNeighbors(n_neighbors=min(k, len(self.embeddings)), metric='cosine')
                self.knn.fit(self.embeddings)
                
            # kneighbors retorna (distancias, indices)
            distances, indices = self.knn.kneighbors([query_vector], n_neighbors=min(k, len(self.embeddings)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                results.append({
                    'metadata': self.metadata[idx],
                    'distance': distances[0][i] # Distância Cosseno (Menor é melhor. 0 = Idêntico)
                })
            return results
        else:
            # Consulta via ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=k
            )
            
            # Normaliza o formato de saída do Chroma para ficar igual ao do Fallback
            parsed_results = []
            if results['metadatas']:
                for i, meta in enumerate(results['metadatas'][0]):
                    parsed_results.append({
                        'metadata': meta,
                        'distance': results['distances'][0][i] if 'distances' in results else 0.0
                    })
            return parsed_results
