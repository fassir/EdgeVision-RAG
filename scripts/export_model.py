from ultralytics import YOLO
import sys

"""
Script de Exportação de Modelo.

Converte o modelo PyTorch (.pt) oficial do YOLOv8 para o formato ONNX.
ONNX é um formato aberto que permite rodar modelos de IA sem depender do PyTorch pesado,
e é otimizado para inferência em CPU e Edge Devices.
"""

def export_model():
    print("Carregando modelo YOLOv8n (Nano)...")
    # Baixa automaticamente o modelo pré-treinado na primeira execução
    model = YOLO("yolov8n.pt") 
    
    print("Exportando para ONNX...")
    # Exporta o modelo
    # opset=12 é uma versão muito estável do padrão ONNX
    path = model.export(format="onnx", opset=12)
    print(f"Modelo exportado com sucesso para: {path}")

if __name__ == "__main__":
    try:
        export_model()
    except Exception as e:
        print(f"Erro durante a exportação: {e}")
        sys.exit(1)
