from fastapi import FastAPI, HTTPException, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
from src.feature_extractor import FeatureExtractor
from src.vector_db import VectorDBWrapper
import time
import io
from PIL import Image

"""
Módulo API (FastAPI).

Este arquivo expõe o motor de busca semântica através de uma interface HTTP REST.
Ele permite que frontends (Web, Mobile, CLI) consultem o banco de dados de vídeo
sem precisar carregar os modelos de IA localmente.

Funcionalidades:
1. Carregamento único dos modelos (Singleton) na inicialização.
2. Monitoramento de métricas (Prometheus).
3. Busca semântica (Texto -> Vídeo).
"""

app = FastAPI(
    title="EdgeVision RAG API", 
    description="API de Busca Semântica em Vídeo. Permite encontrar cenas usando descrições textuais.",
    version="1.0.0"
)

# Instrumenta a API para expor métricas ao Prometheus (latência, contagem de requisições, erros).
# As métricas ficam disponíveis em /metrics
Instrumentator().instrument(app).expose(app)

# Estado Global (Singleton)
# Mantemos essas variáveis globais para carregar os modelos pesados (CLIP, Chroma) apenas UMA vez
# quando a API inicia, e não a cada requisição (o que seria muito lento).
vector_db = None
feature_extractor = None

@app.on_event("startup")
async def startup_event():
    """
    Evento de Inicialização do FastAPI.
    
    Rodado automaticamente quando o servidor começa.
    Responsável por instanciar a rede neural (CLIP) e conectar ao banco de dados.
    """
    global vector_db, feature_extractor
    print("API: Inicializando modelos de IA e Banco de Dados...")
    
    # Carrega o modelo CLIP (Pode demorar alguns segundos)
    feature_extractor = FeatureExtractor()
    
    # Conecta ao ChromaDB ou carrega o Pickle local
    vector_db = VectorDBWrapper()
    
    print("API: Sistema pronto para receber requisições.")

@app.get("/")
def read_root():
    """Rota raiz para verificação de saúde (Health Check)."""
    return {
        "message": "Bem-vindo à EdgeVision RAG API.",
        "usage": "Use GET /search?q='descrição da cena' para buscar segmentos de vídeo."
    }

@app.get("/stats")
def get_stats():
    """
    Retorna estatísticas do sistema.
    
    Útil para dashboards de monitoramento saberem o tamanho do índice.
    """
    count = 0
    if vector_db:
        if vector_db.fallback_mode:
            count = len(vector_db.embeddings) # Contagem direta da lista em memória
        else:
            count = vector_db.collection.count() # Contagem via ChromaDB
            
    return {
        "total_vectors": count,
        "backend_type": "ChromaDB (Robusto)" if not vector_db.fallback_mode else "Local/Sklearn (Leve)",
        "status": "online"
    }

@app.get("/search")
def search(q: str, limit: int = 5):
    """
    Realiza a busca semântica no índice de vídeo.
    
    Fluxo:
    1. Recebe o texto 'q' (ex: "viatura policial").
    2. Converte o texto em um vetor numérico (Embedding) usando CLIP.
    3. Compara esse vetor com todos os vetores armazenados no banco.
    4. Retorna os 'limit' resultados mais próximos (menor distância).
    
    Args:
        q (str): Texto da consulta (Query).
        limit (int): Número máximo de resultados a retornar.
    """
    if not q:
        raise HTTPException(status_code=400, detail="O parâmetro 'q' (query) é obrigatório.")
    
    start_time = time.time()
    
    # 1. Gerar vetor da query (Texto -> Números)
    query_vector = feature_extractor.extract_text_embedding(q)
    
    # 2. Buscar no DB (Cálculo de Distância Cosseno)
    results = vector_db.search(query_vector, k=limit)
    
    # 3. Formatar a resposta para o cliente
    response = []
    for res in results:
        meta = res['metadata']
        # Converte timestamp (segundos totais) para Minutos:Segundos para facilitar leitura
        minutes = int(meta.get('timestamp', 0) // 60)
        seconds = int(meta.get('timestamp', 0) % 60)
        
        response.append({
            "timestamp": meta.get('timestamp'), # Tempo exato no vídeo
            "formatted_time": f"{minutes:02d}:{seconds:02d}", # Formato MM:SS
            "confidence": meta.get('confidence'), # Confiança da detecção original (YOLO)
            "distance": float(res['distance']), # Quão semanticamente próximo é (Menor = Melhor)
            "metadata_completo": meta # Retorna tudo caso o frontend precise
        })
        
    return {
        "query": q,
        "results": response,
        "processing_time_seconds": time.time() - start_time
    }

@app.post("/search_image")
async def search_by_image(file: UploadFile = File(...), limit: int = 5):
    """
    Busca por Imagem (Reverse Image Search).
    
    Permite fazer upload de uma imagem (ex: foto de um objeto ou pessoa)
    e encontrar momentos no vídeo onde algo visualmente similar aparece.
    
    Args:
        file (UploadFile): Arquivo de imagem (JPG/PNG).
        limit (int): Máximo de resultados.
    """
    start_time = time.time()
    
    # 1. Ler arquivo e converter para PIL Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {e}")

    # 2. Gerar Embedding (Imagem -> Números)
    # A classe FeatureExtractor agora aceita PIL images diretamente
    query_vector = feature_extractor.extract_image_embedding(image)
    
    # 3. Buscar no DB
    results = vector_db.search(query_vector, k=limit)
    
    # 4. Formatar Resposta (reutilizando a lógica do search text)
    response = []
    for res in results:
        meta = res['metadata']
        minutes = int(meta.get('timestamp', 0) // 60)
        seconds = int(meta.get('timestamp', 0) % 60)
        
        response.append({
            "timestamp": meta.get('timestamp'),
            "formatted_time": f"{minutes:02d}:{seconds:02d}",
            "confidence": meta.get('confidence'),
            "distance": float(res['distance']),
            "source_video": meta.get('source')
        })
        
    return {
        "query_type": "image",
        "filename": file.filename,
        "results": response,
        "processing_time_seconds": time.time() - start_time
    }
