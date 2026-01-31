import requests
import os

"""
Script de Download de Dados de Amostra.

Baixa automaticamente um vídeo de teste da internet (Pexels) para a pasta `data/`.
Isso permite que o usuário teste o pipeline imediatamente sem precisar fornecer seu próprio vídeo.
"""

def download_file(url, filename):
    """Baixa um arquivo via HTTP com barra de progresso simples (chunks)."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Download concluído: {filename}")
    else:
        print(f"Falha ao baixar de {url}")

def main():
    # Cria a pasta 'data' se não existir
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Fonte alternativa estável para um vídeo de multidão (Pexels - Domínio Público)
    url = "https://videos.pexels.com/video-files/855564/855564-hd_1920_1080_24fps.mp4"
    target_path = os.path.join(data_dir, "people-walking.mp4")
    
    print(f"Baixando vídeo de exemplo para: {target_path}...")
    download_file(url, target_path)

if __name__ == "__main__":
    main()
