import requests
import time
import sys

"""
Script de Teste Automatizado da API.

Realiza chamadas HTTP reais contra o servidor local (localhost:8000) para garantir que
os endpoints principais (Root, Stats, Search) estão funcionando conforme o esperado.
"""

def test_api():
    base_url = "http://localhost:8000"
    print(f"Testando API em {base_url}...")
    
    # 1. Teste de Conectividade (Raiz)
    try:
        resp = requests.get(f"{base_url}/")
        print(f"GET / -> Status: {resp.status_code}")
        print(f"Resposta: {resp.json()}")
    except Exception as e:
        print(f"GET / Falhou: {e} (O servidor está rodando?)")
        return

    # 2. Teste de Estatísticas (Stats)
    try:
        resp = requests.get(f"{base_url}/stats")
        print(f"GET /stats -> Status: {resp.status_code}")
        stats = resp.json()
        print(f"Stats: {stats}")
        
        if stats.get('total_vectors', 0) == 0:
            print("AVISO: O banco de dados está vazio. Rode 'run_pipeline.py' para popular.")
    except Exception as e:
        print(f"GET /stats Falhou: {e}")

    # 3. Teste de Busca (Search)
    query = "person"
    try:
        print(f"Testando busca por '{query}'...")
        resp = requests.get(f"{base_url}/search", params={"q": query})
        print(f"GET /search -> Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            print(f"Encontrou {len(results)} correspondências.")
            if results:
                print(f"Melhor resultado (Timestamp): {results[0].get('timestamp')}s (Distância: {results[0].get('distance'):.4f})")
    except Exception as e:
        print(f"GET /search Falhou: {e}")

if __name__ == "__main__":
    # Espera 2 segundos para dar tempo do servidor iniciar se chamado em script
    time.sleep(2) 
    test_api()
