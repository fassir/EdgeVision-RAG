import requests
import os

"""
Script de Teste para Busca por Imagem.

Este script baixa uma imagem de exemplo (um homem andando) e a envia 
para o novo endpoint /search_image da API.
Serve para verificar se o sistema consegue encontrar pessoas visualmente similares no vídeo.
"""

def test_image_search():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 1. Preparar imagem de query
    # Usando uma foto genérica de pessoa andando para ver se dá match com o vídeo de multidão
    img_url = "https://images.pexels.com/photos/4556773/pexels-photo-4556773.jpeg?auto=compress&cs=tinysrgb&w=400" 
    img_path = os.path.join(data_dir, "query_image.jpg")
    
    if not os.path.exists(img_path):
        print(f"Baixando imagem de exemplo para teste em: {img_path}...")
        try:
            r = requests.get(img_url, timeout=10)
            if r.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Erro ao baixar imagem: {r.status_code}")
                return
        except Exception as e:
            print(f"Erro de conexão: {e}")
            return

    # 2. Chamar a API
    url = "http://localhost:8000/search_image"
    print(f"Enviando '{img_path}' para {url}...")
    
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            # Envia POST multipart/form-data
            r = requests.post(url, files=files)
        
        print(f"Status da Resposta: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print("\n=== RESULTADOS DA BUSCA POR IMAGEM ===")
            print(f"Query: {data.get('filename')}")
            for item in data.get('results', []):
                print(f"- Tempo: {item['formatted_time']} | Distância: {item['distance']:.4f}")
        else:
            print("Erro na API:", r.text)
            
    except Exception as e:
        print(f"Falha ao conectar com a API: {e}")

if __name__ == "__main__":
    test_image_search()