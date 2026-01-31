import cv2
import sys
import os
import time
import uuid
import argparse

# Garante que o diretório 'src' seja visível para importação
# Isso permite que rodemos o script tanto da raiz quanto de subpastas sem erro de ModuleNotFound.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.inference_engine import YOLOv8ONNX
from src.feature_extractor import FeatureExtractor
from src.vector_db import VectorDBWrapper

"""
Pipeline Principal de Processamento de Vídeo.

Este script é o coração do sistema EdgeVision. Ele executa o loop infinito que:
1. Captura quadros do vídeo ou webcam.
2. Detecta objetos (Pessoas) usando YOLOv8.
3. Recorta os objetos detectados.
4. Extrai vetores de características (Embeddings) visuais com CLIP.
5. Indexa esses vetores no Banco de Dados (ex: ChromaDB) junto com o timestamp.

Uso:
    python run_pipeline.py [video_path] [--reset]
"""

def main():
    # Configuração de argumentos via linha de comando
    parser = argparse.ArgumentParser(description="EdgeVision RAG Pipeline")
    parser.add_argument("video_source", nargs="?", default="0", help="Caminho do arquivo de vídeo ou índice da Webcam (padrão: 0)")
    parser.add_argument("--reset", action="store_true", help="Reseta o banco de vetores (Apaga dados antigos)")
    args = parser.parse_args()

    video_source = args.video_source
    # Se o argumento for um dígito (ex: "0" ou "1"), converte para int para abrir a webcam
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
        
    model_path = "yolov8n.onnx"
    
    # Verificação de segurança: Modelo existe?
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo {model_path} não encontrado. Por favor, execute 'python scripts/export_model.py' primeiro.")
        return

    # 1. Inicializa o motor de Inferência (Detecção de Objetos)
    print(f"Inicializando Motor de Inferência (YOLO) com {model_path}...")
    engine = YOLOv8ONNX(model_path, device="cpu") # Força CPU para facilitar demonstração (sem CUDA)

    # 2. Inicializa o Extrator de Características (Visão Computacional Semântica)
    print("Inicializando Extrator de Características (CLIP)...")
    feature_extractor = FeatureExtractor()

    # 3. Inicializa o Banco de Dados Vetorial (Memória do Sistema)
    print(f"Inicializando Vector DB (Reset={args.reset})...")
    vector_db = VectorDBWrapper(reset=args.reset)
    
    # 4. Abre o arquivo de vídeo ou stream
    print(f"Abrindo fonte de vídeo: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo/webcam.")
        return

    print("Pressione 'q' para sair.")
    
    frame_count = 0
    # OTIMIZAÇÃO: Processar embeddings apenas a cada 30 frames (aprox. 1 segundo).
    # Isso evita sobrecarregar o banco com dados repetidos e mantém o FPS alto.
    process_every_n_frames = 30 
    
    # OTIMIZAÇÃO: Configura resolução da câmera para 640x480.
    # Resoluções maiores (HD/4K) deixam o YOLO muito lento na CPU.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Loop Principal (Frame a Frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou falha na leitura.")
            break

        frame_count += 1
        
        # Redimensionamento defensivo: Garante que mesmo vídeos 4K sejam processados rápido
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, 480))
        
        # --- ETAPA DE DETECÇÃO ---
        # Roda o YOLO no frame atual.
        # Retorna o frame desenhado (para display) e a lista de detecções cruas.
        output_frame, detections = engine.infer(frame)

        # --- ETAPA DE INDEXAÇÃO (RAG) ---
        # Só executamos em frames chave (Key Frames) para performance
        if frame_count % process_every_n_frames == 0:
            for (x1, y1, x2, y2, score) in detections:
                # Sanitização de coordenadas: Garante que estão dentro da imagem
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Filtra detecções muito pequenas (ruído)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                
                # Recorta a região de interesse (ROI) - A pessoa detectada
                person_crop = frame[y1:y2, x1:x2]
                
                # Gera o Embedding (Assinatura visual única) do recorte
                embedding = feature_extractor.extract_image_embedding(person_crop)
                
                # Cria metadados e ID único
                record_id = str(uuid.uuid4())
                metadata = {
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, # Tempo em segundos
                    "frame_index": frame_count,
                    "confidence": float(score),
                    "source": str(video_source)
                }
                
                # Salva no Banco de Dados
                vector_db.add(embedding, metadata, record_id)
                
                # Feedback visual temporário (círculo azul) para indicar que salvamos algo
                cv2.circle(output_frame, (x1 + 10, y1 + 10), 5, (255, 0, 0), -1)

        # --- EXIBIÇÃO ---
        # Mostra o tamanho atual do banco na tela
        db_size_text = f"DB Size: {len(vector_db.embeddings) if vector_db.fallback_mode else 'Chroma'}"
        cv2.putText(output_frame, db_size_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("EdgeVision RAG - Detection & Embedding", output_frame)

        # Sai se pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Garante que os dados sejam persistidos no disco ao sair (se estiver em modo arquivo)
    if vector_db.fallback_mode:
        vector_db._save_local_store()
        print("VectorStore local salvo com sucesso.")

if __name__ == "__main__":
    main()
